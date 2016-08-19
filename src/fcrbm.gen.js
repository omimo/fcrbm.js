/*
model: model parameters
genParams: generation parameters
numFrames: number of frames to generate
history: history 
*/
function gen(model, genParams, numFrames, labels, vishistory) {
    console.log("Generating ...");

    timeStamp(1);

    visible     = math.zeros(numFrames + model.nt, model.numdims);
    poshidprobs = math.zeros(numFrames, model.numhid);
    hidstates   = math.zeros(numFrames, model.numhid);

    visible.subset(math.index(math.range(0, model.nt),
                              math.range(0,model.numdims)), 
                   vishistory);

    timeStamp(2);

    for (var f = model.nt; f < numFrames + model.nt; f++ ) {
        console.log(f+': frame '+(f-model.nt +1));
        timeStamp(3);
        // init the visible using the last frame + noise

        var somenoise = math.zeros(1, model.numdims);
        somenoise = somenoise.map(function (value, index, matrix) {
            return math.random() * genParams.initNoise;
        });

        var pastNoise = math.add(visible.subset(math.index(f-1, math.range(0,model.numdims))),
                    somenoise);

        visible.subset(math.index(f,math.range(0,model.numdims)),
                        pastNoise);

        timeStamp(4);

        // build the past by copying from history
        // for hh=nt:-1:1 %note reverse order
        // past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = visible(tt-hh,:);
        var past = math.zeros(1, model.nt * model.numdims);

        for (var hh = model.nt-1; hh >= 0; hh--) {
            var p = math.subset(visible, 
                            math.index(math.range(f-hh,f+1 -hh),
                                        math.range(0,model.numdims)));
            
            past.subset(math.index(math.range(0,1),
                            math.range((model.nt-hh-1)*model.numdims,
                                        (model.nt-hh)*model.numdims) 
                                    ),p);
        }
        timeStamp(5);
        // Calculate the history and features considering factors
        
        // if (lf )
        //     features = labels(tt-lastFrame,:)*labelfeat;
        // else
        //     features = labels(tt-lastFrame,:);
        // end
        var numLabels = math.size(labels).get([1]);
        var ll = math.subset(labels,
                        math.index(math.range(f-model.nt,f-model.nt+1),
                                    math.range(0,numLabels)));

        if (model.lf == 1)
            features =  math.multiply(ll, model.labelfeat);
        else
            features = ll;

        // %undirected model
        // yfeat = features*featfac;
        
        var yfeat = math.multiply(features, model.featfac);

        // %autoregressive model
        // ypastA = past*pastfacA;
        // yfeatA = features*featfac;
        
        var ypastA = math.multiply(past, model.pastfacA);
        var yfeatA = math.multiply(features, model.featfac);

        // %directed vis-hid model
        // ypastB = past*pastfacB;
        // yfeatB = features*featfac;

        var ypastB = math.multiply(past, model.pastfacB);
        var yfeatB = math.multiply(features, model.featfac); 
        timeStamp(6);
        // %constant term during inference
        // %(not dependent on visibles)
        // constinf = -(ypastB.*yfeatB)*hidfacB' - hidbiases;
        
        var t = math.dotMultiply(ypastB,yfeatB);
        t = math.multiply(-1,t);
        t = math.multiply(t,math.transpose(model.hidfacB));
        var hidbiases = math.matrix(model.hidbiases).clone();
        hidbiases.resize([1,model.hidbiases.length]);
        var constinf =  math.subtract(t,hidbiases);
        timeStamp(7);
        // %constant term during reconstruction
        // %(not dependent on hiddens)
        // constrecon = (yfeatA.*ypastA)*visfacA' + visbiases;

        var t2 = math.dotMultiply(yfeatA,ypastA);
        t2 = math.multiply(t2,math.transpose(model.visfacA));
        var visbiases = math.matrix(model.visbiases).clone();
        visbiases.resize([1,model.visbiases.length]);
        var constrecon =  math.add(t2,visbiases);

        // Gibbs sampling
        
        timeStamp(8);
        for (var g = 0; g < genParams.numGibbs; g++) {
            // yvis = visible(tt,:)*visfac;        
            var yvis = math.multiply(
                visible.subset(math.index(f,math.range(0,model.numdims))),
                model.visfac);

            // %pass through sigmoid
            // %only part from "undirected" model changes
            // poshidprobs(tt,:) = 1./(1 + exp(-(yvis.*yfeat)*hidfac' + constinf));

            var bottomup = math.dotMultiply(yvis,yfeat);
                bottomup = math.multiply(-1,bottomup);
                bottomup = math.multiply(bottomup,math.transpose(model.hidfac));

            var et = math.add(1,math.exp(math.add(bottomup,constinf)));
            et = math.dotDivide(1,et);
            poshidprobs.subset(math.index(f+model.nt,math.range(0,model.numhid)),et);

            // %Activate the hidden units
            // hidstates(tt,:) = single(poshidprobs(tt,:) > rand(1,numhid));
            
            var act = math.zeros(1, model.numhid);
            var pp = poshidprobs.subset(math.index(f+model.nt,math.range(0,model.numhid)));        
            act = pp.clone().map(function (value, index, matrix) {
                if (value > math.random())
                    return 1;
                else
                    return 0;            
            });        
            hidstates.subset(math.index(f+model.nt,math.range(0,model.numhid)),act);
        

            // yhid = hidstates(tt,:)*hidfac;
            var yhid = math.multiply(act, model.hidfac);

            // %NEGATIVE PHASE
            // %Don't add noise at visibles
            // %Note only the "undirected" term changes
            // visible(tt,:) = (yfeat.*yhid)*visfac' + constrecon;

            var v = math.dotMultiply(yfeat, yhid);
            v = math.multiply(v, math.transpose(model.visfac));
            v = math.add(v, constrecon);
            visible.subset(math.index(f,math.range(0,model.numdims)),
                        v);
        timeStamp(9);
        }
        
        var pp = poshidprobs.subset(math.index(f+model.nt,math.range(0,model.numhid)));             

        // Meanfield
        // yhid_ = poshidprobs(tt,:)*hidfac; %smoothed version
        var yhid_ = math.multiply(pp, model.hidfac);

        // Approximation 
        // visible(tt,:) = (yfeat.*yhid_)*visfac' + constrecon;    
        var v = math.dotMultiply(yfeat, yhid_);
            v = math.multiply(v, math.transpose(model.visfac));
            v = math.add(v, constrecon);
        visible.subset(math.index(f,math.range(0,model.numdims)),
                        v);

        updateHidAc2(f,pp.toArray());
    }

    timeStamp(10);
    console.log("Done generating!");
}


module.exports = fcrbm.gen;