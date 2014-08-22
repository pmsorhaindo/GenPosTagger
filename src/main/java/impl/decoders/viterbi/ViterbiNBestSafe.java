package impl.decoders.viterbi;

import edu.berkeley.nlp.util.ArrayUtil;
import impl.Model;
import impl.ModelSentence;
import util.Util;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by ps324 on 19/08/2014.
 */
public class ViterbiNBestSafe {

    private Model m;
    private int n;
    private int numLabels;
    private Util u;
    private ModelSentence sentence;
    private ArrayList<Sequence> exclusionList = new ArrayList<>();

    public ViterbiNBestSafe(Model m){

        u = new Util();
        this.n =2;
        this.m = m;
        this.numLabels = m.labelVocab.size();
        assert numLabels != 0;

    }

    private Sequence computeCandidates(ModelSentence sentence, Sequence maxSeq) {
        //Initialisation
        int T = sentence.T; // Number of tokens to be tagged.
        ArrayList<Sequence> inclusionList = new ArrayList<>();
        exclusionList.addAll(maxSeq.getPathSegments());

        while (inclusionList.size()<=0 && inclusionList.get(0).size() > T) {
            // Check all probs aren't already calculated
            calculateCandiateSubset(sentence, exclusionList,inclusionList,maxSeq);
            //System.out.println("*** new sequence "+seq.getListOfNodes().toString()+" : "+seq.getProbabilityOfSequence());
        }

        int bestSeqIndex = 0;

        for  (int z = 0; z<inclusionList.size(); z++){

            double zProb = inclusionList.get(z).getProbabilityOfSequence();
            double bestProb = inclusionList.get(bestSeqIndex).getProbabilityOfSequence();

            int bestSize = inclusionList.get(bestSeqIndex).getListOfNodes().size();
            int zSize = inclusionList.get(z).getListOfNodes().size();

            if(zSize == T+1) {

                boolean cond1 = (zProb > bestProb);
                boolean cond2 = bestSize < T+1;
                if(cond1 || cond2)
                {
                    //System.out.println("change! "+ inclusionList.get(z).getListOfNodes().size());
                    bestSeqIndex = z;
                }
            }
        }
        System.out.println("***");
        for(int i=0;i<inclusionList.size();i++)
        {
            System.out.println(inclusionList.get(i).toString());
        }
        System.out.println("***");
        inclusionList.get(bestSeqIndex).generateSegments();
        exclusionList.add(inclusionList.get(bestSeqIndex));
        //System.out.println("BEST SEQ:"+ T+ " "+inclusionList.get(bestSeqIndex).getListOfNodes().toString());
        //System.out.println("BEST SEQ:"+ inclusionList.get(bestSeqIndex).getListOfNodes().size());
        return inclusionList.get(bestSeqIndex);
    }

    private void calculateCandiateSubset(ModelSentence sentence, ArrayList<Sequence> exclusionList, ArrayList<Sequence> inclusionList, Sequence maxSeq) {
        if(inclusionList.isEmpty()||inclusionList.size()<=0){
            ArrayList<Sequence> tempInclusionList = new ArrayList<>();
            {
                double[] labelScores = new double[numLabels];
                double[] prevcurr = new double[m.numLabels];
                computeVitLabelScores(0, m.startMarker(), sentence, prevcurr);
                //System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
                ArrayUtil.logNormalize(prevcurr);
                double[] newProb = ArrayUtil.add(prevcurr, labelScores);

                for (int i = 0; i < this.numLabels; i++) {
                    ArrayList<Integer> nodeStub = new ArrayList<>();
                    nodeStub.add(m.startMarker());
                    nodeStub.add(i);

                    Sequence newS = new Sequence(newProb[i], nodeStub);
                    if (!exclusionList.contains(newS)) {
                        //System.out.println("Sequence Added "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
                        tempInclusionList.add(newS);
                    } else {
                        //System.out.println("Sequence Barred! "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
                    }
                }
            }

            for (int i = 0; i < exclusionList.size() - 1; i++) {
                Sequence s = exclusionList.get(i);
                double[] labelScores = new double[numLabels];
                double[] prevcurr = new double[m.numLabels];
                if (s.size() - 1 >= sentence.T) {
                    continue;
                }
                computeVitLabelScores((s.size() - 1), s.get(s.size() - 1), sentence, prevcurr);
                //System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
                ArrayUtil.logNormalize(prevcurr);
                double[] newProb = ArrayUtil.add(prevcurr, labelScores);

                for (int j = 0; j < this.numLabels; j++) {
                    ArrayList<Integer> nodeStub = new ArrayList<>();
                    nodeStub.addAll(s.getListOfNodes());
                    nodeStub.add(j);

                    Sequence newS = new Sequence(newProb[j], nodeStub);
                    if (!exclusionList.contains(newS)||!inclusionList.contains(newS)) {
                        //System.out.println("Sequence Added "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
                        tempInclusionList.add(newS);
                    } else {
                        //System.out.println("Sequence Barred! "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
                    }
                }
            }

            int bestSeqIndex = 0;
            for (int i = 0; i < tempInclusionList.size(); i++) {
                double iProb = tempInclusionList.get(i).getProbabilityOfSequence();
                double bestProb = tempInclusionList.get(i).getProbabilityOfSequence();
                if (iProb >= bestProb) {
                    bestSeqIndex = i;
                }
            }
            inclusionList.add(tempInclusionList.get(bestSeqIndex));
        }
    }


    //TODO remove
    public void computeVitLabelScores(int t, int prior, ModelSentence sentence, double[] labelScores) {
        Arrays.fill(labelScores, 0);
        m.computeBiasScores(labelScores);
        //System.out.println("prior = " + prior);
        viterbiEdgeScores(prior, sentence, labelScores);
        //System.out.println("t = " + t);
        m.computeObservedFeatureScores(t, sentence, labelScores);
    }

    /**
     * @return dim T array s.t. labelScores[t]+=score of label prior followed by
     *         label t
     **/
    public void viterbiEdgeScores(int prior, ModelSentence sentence,
                                  double[] EdgeScores) {
        for (int k = 0; k < numLabels; k++) {
            EdgeScores[k] += m.edgeCoefs[prior][k];
        }
    }

}
