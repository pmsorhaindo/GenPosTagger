package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import util.Util;
import edu.berkeley.nlp.util.ArrayUtil;

public class ViterbiArray implements IDecoder {
	
	private Model m;
	private int numLabels;
	private int N;
	private Util u;
	
	public ViterbiArray(Model m){
		
		this.N = 4;
		this.u = new Util();
		this.m = m;
		this.numLabels = m.labelVocab.size();
		assert numLabels != 0;
		
	}

	@Override
	public void decode(ModelSentence sentence) {
		viterbiArrayDecode(sentence);

	}

	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public void viterbiArrayDecode(ModelSentence sentence) {
		// Initialization
		int T = sentence.T; // Number of tokens to be tagged.
		int branchLimit = 4;
		sentence.labels = new int[T]; // final labeled sequence
		double[][] vit = new double[1][numLabels]; // viterbi Matrix
		double[][] labelScores = new double[branchLimit][numLabels]; 
		
		computeVitLabelScores(0, m.startMarker(), sentence, labelScores[0]);
		ArrayUtil.logNormalize(labelScores[0]);

        System.out.println("Nooo@!!");
        // Assigning first label scores to the first column in the viterbi matrix
		//vit[0] = labelScores;
		
	}
	
	
	
	// TODO remove
	public void computeVitLabelScores(int t, int prior, ModelSentence sentence,
			double[] labelScores) {
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
