package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.DecoderUtils;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import util.Util;
import edu.berkeley.nlp.util.ArrayUtil;

public class Viterbi implements IDecoder {

	private int numLabels;
	private Model m;
	private DecoderUtils dUtils;
	private Util u;
	private ArrayList<Double> probs;
	
	
	public Viterbi(Model m){
		
		numLabels = m.labelVocab.size();
		this.m = m;
		this.dUtils = new DecoderUtils(m);
		this.u = new Util();
	}
	
	@Override
	public void decode(ModelSentence sentence) {
		viterbiDecode(sentence);

	}

	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * vit[t][k] is the max probability such that the sequence from 0 to t has
	 * token t labeled with tag k. (0<=t<T) bptr[t][k] gives the max prob. tag
	 * of token t-1 (t=0->startMarker)
	 */
	public void viterbiDecode(ModelSentence sentence) {
		//Initialisation
		int T = sentence.T; // Number of tokens to be tagged.
		sentence.labels = new int[T]; // final labeled sequence
		int[][] bptr = new int[T][numLabels]; // backpointer array
		double[][] vit = new double[T][numLabels]; // viterbi Matrix
		double[] labelScores = new double[numLabels]; 

		
		computeVitLabelScores(0, m.startMarker(), sentence, labelScores);
		ArrayUtil.logNormalize(labelScores);
		
		// Assigning first label scores to the first column in the viterbi matrix
		vit[0] = labelScores;
		//u.p("first vit:");
		//u.p(vit[0]);
		
		for (int k = 0; k < numLabels; k++) {
			// start marker for all labels
			bptr[0][k] = m.startMarker();
		}
		u.p(bptr[0]);

		// Calculate viterbi label scores.
		for (int t = 1; t < T; t++) {
			double[][] prevcurr = new double[numLabels][numLabels];
			for (int s = 0; s < numLabels; s++) {
				//System.out.println("labelScores[" + s + "]" + labelScores[s]);
				computeVitLabelScores(t, s, sentence, prevcurr[s]);
				//System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
				ArrayUtil.logNormalize(prevcurr[s]);
				prevcurr[s] = ArrayUtil.add(prevcurr[s], labelScores[s]);
			}
			//System.out.println("prevCurr "+t+":");
			//u.p(prevcurr);
			for (int s = 0; s < numLabels; s++) {
				double[] sprobs = u.getColumn(prevcurr, s);
				bptr[t][s] = ArrayUtil.argmax(sprobs); // u.nthLargest(2, sprobs);
				vit[t][s] = sprobs[bptr[t][s]];
			}
			System.out.println("bptr "+t+":");
			u.p(bptr[t]);
			labelScores = vit[t];
            System.out.println("vit "+t+":");
            u.p(vit[t]);
        }
		
		this.probs = new ArrayList<>();
		sentence.labels[T - 1] = ArrayUtil.argmax(vit[T - 1]);
		System.out.print("***" + m.labelVocab.name(sentence.labels[T - 1]));
		double prob = vit[T - 1][sentence.labels[T - 1]]; //Math.exp(vit[T - 1][sentence.labels[T - 1]]);
		System.out.println(" with prob: " + prob);
		this.probs.add(prob);
		
		int backtrace = bptr[T - 1][sentence.labels[T - 1]];
		for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
			sentence.labels[i] = backtrace;
			double newProb = vit[i][backtrace]; //Math.exp(vit[i][backtrace]);
			System.out.println("***" + m.labelVocab.name(backtrace)
					+ " with prob: " + newProb);
			this.probs.add(newProb);
			backtrace = bptr[i][backtrace];
		}
		Collections.reverse(this.probs);

		assert (backtrace == m.startMarker());
		u.p("label sequence");
		u.p(sentence.labels);
	}
	
	public void computeVitLabelScores(int t, int prior, ModelSentence sentence,
			double[] labelScores) {
		Arrays.fill(labelScores, 0);
		dUtils.computeBiasScores(labelScores);
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

	public ArrayList<Double> getProbs() {
		return probs;
	}

	public void setProbs(ArrayList<Double> probs) {
		this.probs = probs;
	}
}
