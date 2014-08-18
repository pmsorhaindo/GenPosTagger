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
		this.n = 27;
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
			x = computeCandidates(sentence, s);
			sentence.labels = x.getLabelIndexes();
			//System.out.println("3rd plus:");
			//u.p(sentence.labels);
			paths[i] = sentence.labels;
            confs[i] = x.getProbabilityOfSequence();
			//u.p(paths);
			s = x;

		}

		sentence.nPaths= paths;
        sentence.confidences = confs;
		 
	}

	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	
	private Sequence computeCandidates(ModelSentence sentence, Sequence maxSeq) {
		//Initialisation
		int T = sentence.T; // Number of tokens to be tagged.
		ArrayList<Sequence> inclusionList = new ArrayList<>();
		exclusionList.addAll(maxSeq.getPathSegments());
		
		for (int i=0; i<T; i++)
		{
			// Check all probs aren't already calculated
			Sequence seq = new Sequence(); 
			seq = calculateCandiateSubset(sentence, i, exclusionList,inclusionList,maxSeq);
			//System.out.println("*** new sequence "+seq.getListOfNodes().toString()+" : "+seq.getProbabilityOfSequence());
		}
		
		int bestSeqIndex = 0;
		for  (int z = 0; z<inclusionList.size(); z++){
			if(inclusionList.get(z).getListOfNodes().size()== T+1) {
				
				if((inclusionList.get(z).getProbabilityOfSequence() > inclusionList.get(bestSeqIndex).getProbabilityOfSequence()) || inclusionList.get(bestSeqIndex).getListOfNodes().size() < T+1) 
					//TODO length check to prevent problems when n gets larger. Done?
				{
					bestSeqIndex = z;
				}
			}
		}
		inclusionList.get(bestSeqIndex).generateSegments();
		exclusionList.add(inclusionList.get(bestSeqIndex));
		System.out.println("i:"+ T+ " "+inclusionList.get(bestSeqIndex).getListOfNodes().toString());
		return inclusionList.get(bestSeqIndex);
	}
	
	private Sequence calculateCandiateSubset(ModelSentence sentence, int token, ArrayList<Sequence> exclusionList, ArrayList<Sequence> inclusionList, Sequence maxSeq) {
		
		for (int t = 0; t< (sentence.T); t++)
		{
			//System.out.println("iths " + t +" :"+maxSeq.getPathSegments().size());
			Sequence s = maxSeq.getIthPathSegment(t);
			
			double[] prevcurr = new double[m.numLabels];
			computeVitLabelScores(t, maxSeq.getListOfNodes().get(t), sentence, prevcurr);
			//System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
			ArrayUtil.logNormalize(prevcurr);
			double[] newProb = ArrayUtil.add(prevcurr, s.getProbabilityOfSequence());
			
			for(int i = 0; i<this.numLabels; i++)
			{
				ArrayList<Integer> nodeStub = new ArrayList<>(); 
				nodeStub.addAll(s.getListOfNodes());
				nodeStub.remove(nodeStub.size()-1);
				//calculate all i to i+1 sequences excluding those comprising the exclusionlist
				
				nodeStub.add(i);
				Sequence newS = new Sequence(newProb[i],nodeStub);
				
				if(!exclusionList.contains(newS))
				{
					//System.out.println("Sequence Added "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
					inclusionList.add(newS);
				}
				else
				{
					//System.out.println("Sequence Barred! "+newS.getListOfNodes().toString()+" : "+newS.getProbabilityOfSequence());
				}
				
			}
		}
		
		int bestSeqIndex = 0;
		for  (int z = 0; z<inclusionList.size(); z++){
			if(inclusionList.get(z).getProbabilityOfSequence() > inclusionList.get(bestSeqIndex).getProbabilityOfSequence() || z == 0)
			{
				bestSeqIndex = z;
			}
		}
		return inclusionList.get(bestSeqIndex);
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
