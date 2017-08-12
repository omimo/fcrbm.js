//
importScripts("../node_modules/linear-algebra/dist/linear-algebra.min.js");
//

var inputModel; 
var tm; 

var genParams = {
    numGibbs: 4,
    initNoise: 0.05,
};

var visible;

var linearAlgebra =  linearAlgebra(),
    Vector = linearAlgebra.Vector,
    Matrix = linearAlgebra.Matrix;


onmessage = function(e) {
  console.log('Message received from main script');
  generate(e.data[0], e.data[1], e.data[2], e.data[3], e.data[4]);
};

var lastTime = 0;
var timeCount = 0;
function timeStamp() {
    return;
    if (lastTime == 0) {
        lastTime = Date.now();                
    }
    timeCount++;
    console.log(timeCount +": " + (Date.now()-lastTime));
    lastTime = Date.now();
}

/*
model: model parameters
genParams: generation parameters
numFrames: number of frames to generate
history: history 
*/
function generate(model, genParams, numFrames, labels, vishistory) {
    console.log("Generating ...");

    timeStamp(1);

    visible     = Matrix.zero(numFrames + model.nt, model.numdims);
    poshidprobs = Matrix.zero(numFrames, model.numhid);
    hidstates   = Matrix.zero(numFrames, model.numhid);

    for(i=0;i<vishistory.data.length;i++) {
        visible.data[i] = vishistory.data[i];
    }

    // de-objectifying and inlining everything
    numLabels = labels.data[0].length;

    labelfeat   = new Matrix(model.labelfeat);
    featfac     = new Matrix(model.featfac);
    pastfacA    = new Matrix(model.pastfacA);
    pastfacB    = new Matrix(model.pastfacB);
    hidfacB     = new Matrix(model.hidfacB);
    hidfac      = new Matrix(model.hidfac);
    nt          = model.nt;
    numdims     = model.numdims;
    numhid      = model.numhid;
    hidbiases   = new Matrix(model.hidbiases);
    visfac      = new Matrix(model.visfac);
    visfacA     = new Matrix(model.visfacA);
    visbiases   = new Matrix(model.visbiases);
    lf          = model.lf;

    initNoise   = genParams.initNoise;

    timeStamp(2);

    for (var f = nt; f < numFrames + nt; f++ ) {        
        console.log(f+': frame '+(f-nt +1));
        timeStamp(3);
        // init the visible using the last frame + noise    
        visible.data[f] = (new Matrix(visible.data[f-1]).plusEach(Math.random() * initNoise)).data[0];

        timeStamp(4);

        // build the past by copying from history
        // for hh=nt:-1:1 %note reverse order
        // past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = visible(tt-hh,:);
        past = Matrix.zero(1 , nt * numdims);
        
        for (var hh = nt-1; hh >= 0; hh--) {            
            for (i=numdims*(nt-hh-1);i<numdims*(nt-hh);i++)
                for (j=0;j<numdims;j++)
                    past.data[0][i+j] = visible.data[f-hh-1][j]; 
        }        
        timeStamp(5);
        // Calculate the history and features considering factors        
        var features = new Matrix(labels.data[f-nt]);        
        if (lf == 1)
            features.dot_(labelfeat);        

        timeStamp(6);
        // %undirected model
        // yfeat = features*featfac;        
        var yfeat = features.dot(featfac);

        // %autoregressive model
        // ypastA = past*pastfacA;
        // yfeatA = features*featfac;        
        var ypastA = past.dot(pastfacA);
        var yfeatA = features.dot(featfac);

        // %directed vis-hid model
        // ypastB = past*pastfacB;
        // yfeatB = features*featfac;
        var ypastB = past.dot(pastfacB);
        var yfeatB = features.dot(featfac);

        timeStamp(7);
        // %constant term during inference
        // %(not dependent on visibles)
        // constinf = -(ypastB.*yfeatB)*hidfacB' - hidbiases;        
        var t = ypastB.mul(yfeatB);
        t.mulEach_(-1);
        var constinf = t.dot(hidfacB.trans());
        constinf.minus_(hidbiases);
        timeStamp(8);

        // %constant term during reconstruction
        // %(not dependent on hiddens)
        // constrecon = (yfeatA.*ypastA)*visfacA' + visbiases;
        var t2 = yfeatA.mul(ypastA);        
        var constrecon = t2.dot(visfacA.trans());
        // var visbiases2 = Matrix.reshapeFrom(visbiases,1,visbiases.length);
        constrecon.plus_(visbiases);            
        
        timeStamp(9);
        
        // Gibbs sampling
        for (var g = 0; g < genParams.numGibbs; g++) {
            // yvis = visible(tt,:)*visfac;                              
            var yvis = new Matrix(visible.data[f]);             
            yvis.dot_(visfac);

            // %pass through sigmoid
            // %only part from "undirected" model changes
            // poshidprobs(tt,:) = 1./(1 + exp(-(yvis.*yfeat)*hidfac' + constinf));

            var bottomup = yvis.mul(yfeat);
                // bottomup.mulEach_(-1);
            var et = bottomup.dot(hidfac.trans());

            et.minus_(constinf);
            // et.sigmoid_();
            // console.log(et.data[0]);
            et.map_(function(v) {
                return 1.0/(1+Math.exp(-v));    
            });
            // console.log(et.data[0]);           
            poshidprobs.data[f] = et.data[0];                        
            // %Activate the hidden units
            // hidstates(tt,:) = single(poshidprobs(tt,:) > rand(1,numhid));
                        
            var act = new Matrix(poshidprobs.data[f]);
            
            act.map_(function(v) {
                if (v > Math.random())
                    return 1;
                else
                    return 0;
            });
            // for (i=0;i<act.length;i++)
            // console.log(act.data[0]);
            hidstates.data[f] = act.data[0];

            // yhid = hidstates(tt,:)*hidfac;            
            var yhid = act.dot(hidfac);

            // %NEGATIVE PHASE
            // %Don't add noise at visibles
            // %Note only the "undirected" term changes
            // visible(tt,:) = (yfeat.*yhid)*visfac' + constrecon;

            var v = yfeat.mul(yhid);            
            var vv = v.dot(visfac.trans());                        
            vv.plus_(constrecon);            
            visible.data[f] = vv.data[0];

            // console.log(visible.data[f]);
            
        timeStamp(10);
        }                             
        var yhid_ = new Matrix(poshidprobs.data[f]);
        // Meanfield
        // yhid_ = poshidprobs(tt,:)*hidfac; %smoothed version        
        yhid_.dot_(hidfac);

        // Approximation 
        // visible(tt,:) = (yfeat.*yhid_)*visfac' + constrecon;    
        
        var v = yfeat.mul(yhid_).dot(visfac.trans()).plus(constrecon);
        visible.data[f] = v.data[0];
        
        // updateHidAc2(f,poshidprobs.data[f]);
        postMessage([f,poshidprobs.data[f]]);
        timeStamp(11);
    }

    timeStamp(12);
    console.log("Done generating!");
}