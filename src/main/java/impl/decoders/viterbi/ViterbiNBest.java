package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;

import edu.berkeley.nlp.util.ArrayUtil;
import util.Util;

public class ViterbiNBest implements IDecoder {
	
	private Model m;
	private int n;
	private int numLabels;
	private Util u;
	private ModelSentence sentence;
	private ArrayList<Sequence> exclusionList = new ArrayList<>();
	
	public ViterbiNBest(Model m){
		
		u = new Util();
		this.n =2;
		this.m = m;
		this.numLabels = m.labelVocab.size();
		assert numLabels != 0;
		
	}

	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	@Override
	public void decode(ModelSentence sentence) {
		this.sentence = sentence;
		Viterbi vit = new Viterbi(m);
		vit.decode(sentence);
		//u.p(sentence.labels);
		ArrayList<Double> sequenceProbs = vit.getProbs();
		//System.out.println("pre viterbi complete!");
		
		int[][] paths = new int[n][sentence.T];
        double[] confs = new double[n];
		Sequence maxSeq = new Sequence(m.startMarker(), sequenceProbs,sentence.labels);
		paths[0] = sentence.labels;
        Sequence s = new Sequence();

        if(this.n > 1) {
            s = computeCandidates(sentence, maxSeq);
            sentence.labels = s.getLabelIndexes();
            //u.p(sentence.labels);
            paths[1] = sentence.labels;
        }

        if(this.n > 2)
		for(int i=2; i<this.n; i++)
		{
			Sequence x = new Sequence();
            u.p(s.toString());
			x = computeCandidates(sentence, s);
			sentence.labels = x.getLabelIndexes();
			//System.out.println("3rd plus:");
			//u.p(sentence.labels);
			paths[i] = sentence.labels;
            confs[i] = x.getProbabilityOfSequence();
			//u.p(paths);
			s = x;

		}

		sentence.nPaths = paths;
        sentence.pathConfidences = confs;
	}

    public Sequence computeCandidates(ModelSentence sentence, Sequence maxSeq) {

        int T = sentence.T; // Number of tokens to be tagged.
        ArrayList<Sequence> inclusionList = new ArrayList<>();
        boolean explored = false;

        while(!explored)
        {
            explored = calculateCandiateSubset(sentence, exclusionList,inclusionList,maxSeq);
        }

        return null;
    }

    private boolean calculateCandiateSubset(ModelSentence sentence, ArrayList<Sequence> exclusionList, ArrayList<Sequence> inclusionList, Sequence maxSeq) {

        boolean explored = false;

        if(inclusionList.isEmpty())
        {
            caluclateBaseInclusionList(inclusionList);
        }
        else{
            explored = furtherExploreInclusionList(inclusionList);
        }
        return explored;
    }

    private void caluclateBaseInclusionList(ArrayList<Sequence> inclusionList){

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

    private boolean furtherExploreInclusionList(ArrayList<Sequence> inclusionList){
        ArrayList<Sequence> tempInclusionList = new ArrayList<>();

        for (int i = 0; i < inclusionList.size(); i++) {
            Sequence s = inclusionList.get(i);
            double[] labelScores = new double[numLabels];
            double[] prevcurr = new double[m.numLabels];
            if (s.size() - 1 >= sentence.T) {
                continue;
            }
            computeVitLabelScores((s.size() - 1), s.get(s.size() - 1), sentence, prevcurr);
            // System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
            ArrayUtil.logNormalize(prevcurr);
            double[] newProb = ArrayUtil.add(prevcurr, labelScores);

            for (int j = 0; j < this.numLabels; j++) {
                ArrayList<Integer> nodeStub = new ArrayList<>();
                nodeStub.addAll(s.getListOfNodes());
                nodeStub.add(j);

                Sequence newS = new Sequence(newProb[j], nodeStub);
                System.out.println(newS.toString());
                if (!exclusionList.contains(newS)) {
                    System.out.println("Sequence Added "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
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
        if (tempInclusionList.get(bestSeqIndex).size() == sentence.T) {
            System.out.println("****"+tempInclusionList.get(bestSeqIndex).toString());
            return true;
        }
        else {
            return false;
        }
    }


	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	


	public void viterbiNBest(ModelSentence sentence) {
		// TODO refactor in here
		//int T = sentence.T;
		//sentence.labels = new int[T];
		//for (int i = 0; i<T; i++) sentence.labels[i]=i;
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
