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

public class ViterbiArrayThird implements IDecoder {
	
	private Model m;
	private int numLabels;
	private int N;
	private Util u;
	
	public ViterbiArrayThird(Model m){
		
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
		sentence.labels = new int[T]; // final labeled sequence
		int[][][] bptr = new int [N][T][numLabels]; // backpointer array
		int[][][] bptrV = new int [N][T][numLabels];
		double[][][] vit = new double[N][T][numLabels]; // viterbi Matrix
		double[][] labelScores = new double[N][numLabels];
		
		for(int i = 0; i<N; i++) {
			computeVitLabelScores(0, m.startMarker(), sentence, labelScores[i]);
			ArrayUtil.logNormalize(labelScores[i]);
			//Assigning first label scores to the first column in the viterbi matrix
			vit[i] = labelScores;
			u.p("first vit:");
			u.p(vit[0][0]);
		}
		
		// Assigning the start marker to the first position.
		for(int i = 0; i<N; i++)
		{
			for (int k = 0; k < numLabels; k++) {
				// start marker for all labels
				bptr[i][0][k] = m.startMarker();
			}
			u.p("first bptr "+i+":");
			u.p(bptr[i]);
		}
		
		// Calculate viterbi label scores.
		for (int t = 1; t < T; t++) {
			for (int n = 0; n<N; n++) {
				double[][] prevcurr = new double[numLabels][numLabels];
				for (int s = 0; s < numLabels; s++) {
					//System.out.println("labelScores[" + s + "]" + labelScores[s]);
					computeVitLabelScores(t, s, sentence, prevcurr[s]);
					//System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
					ArrayUtil.logNormalize(prevcurr[s]);
					prevcurr[s] = ArrayUtil.add(prevcurr[s], labelScores[n]);
				}

				for (int s = 0; s < numLabels; s++) {
					double[] sprobs = u.getColumn(prevcurr, s);
					bptr[n][t][s] = u.nthLargest((n+1), sprobs);
					System.out.println("n:"+n+" t:"+t+" T:"+T+" s:"+s+" bptr[n][t][s]:"+bptr[n][t][s]);
					System.out.println("sprobs[bptr[n][t][s]]:"+sprobs[bptr[n][t][s]]);
					vit[n][t][s] = sprobs[bptr[n][t][s]];
				}
			}
			System.out.println("bptr "+t+":");
			u.p(bptr[0]);
			for(int n = 0; n<N; n++)
			{
				labelScores[n] = vit[n][t];
			}
			
		}
		
		
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
