package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.DecoderUtils;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import util.Util;
import util.ViterbiPaths;
import edu.berkeley.nlp.util.ArrayUtil;

public class ViterbiDiverge implements IDecoder {
	
	private int numLabels;
	private Util u;
	private DecoderUtils dUtils;
	private Model m;
    public int n = 1;
	
	public ViterbiDiverge(Model m, int returnK) {
		
		numLabels = m.labelVocab.size(); // TODO initialize numLabels properly.
		assert numLabels != 0;
		dUtils = new DecoderUtils(m);
		u = new Util();
        n = returnK;
		this.m = m;
	}

	@Override
	public void decode(ModelSentence sentence) {
        sentence.K = this.n;
		splitViterbiDecode(sentence);
	}

	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * Implement several runs of viterbi but adding a deviation on each each
	 * step. then on each of the runs that splits switching back to finding the
	 * maximum path. Is this cool?
	 * @return 
	 */
	public void splitViterbiDecode(ModelSentence sentence) {
		
		ViterbiPaths vp = new ViterbiPaths();
		for(int i=0; i<sentence.T; i++)
		{
			divergedViterbiDecode(sentence, i, vp);
			//System.out.println("paths size: "+vp.getPaths().size());
			//System.out.println("probs size: "+vp.getProbs().size());
		}
		Double[] objArr = (Double[]) (vp.getProbs()).toArray(new Double[0]);
		double[] arr = ArrayUtils.toPrimitive(objArr);
		int index = u.nthLargest(1, arr);
		//System.out.println("index of largest: "+index +":"+objArr.length+":"+arr.length);
		//System.out.println("probs: ");
		//Util.p(arr);
		
		ArrayList<Integer> vPathArrList = vp.getPaths().get(index);
		Integer[] vPathArrListObjArr = (Integer[]) vPathArrList.toArray(new Integer[0]);

        //for(Integer x : vPathArrList) System.out.println(x);


		int[] vPath = ArrayUtils.toPrimitive(vPathArrListObjArr);

        int[][] npaths = vp.topNHighestPaths(this.n, sentence.T);
        double[][] confs = vp.topNHighestConfidences(this.n, sentence.T);
        double [] pConfs = vp.topNHighestProbabilities(this.n);
		sentence.nPaths = npaths;
        sentence.confidences = confs;
        sentence.pathConfidences = pConfs;
		//System.out.println("n("+n+") paths:");
		//Util.p(npaths);
        //Util.p(arr);
		sentence.labels = vPath;
	}

	
	public ViterbiPaths divergedViterbiDecode(ModelSentence sentence, int divergePoint, ViterbiPaths vp) {
		int T = sentence.T;
		sentence.labels = new int[T];
		int[][] bptr = new int[T][numLabels];
		double[][] vit = new double[T][numLabels];
		double[] labelScores = new double[numLabels];

		// System.out.println("start Marker = "+ m.startMarker());
		computeVitLabelScores(0, m.startMarker(), sentence, labelScores);
		// System.out.println("label score init: \n" + priArr(labelScores));
		ArrayUtil.logNormalize(labelScores);
		// System.out.println("label score init (log norm'd): \n" +
		// priArr(labelScores));

		// initialization
		vit[0] = labelScores;

		for (int k = 0; k < numLabels; k++) {
			// start marker for all labels
			bptr[0][k] = m.startMarker();
		}

		// System.out.println("Initial back tagPointer array: " + priArr(bptr[0]));

		// Calculate viterbi scores
		for (int t = 1; t < T; t++) {
			//System.out.println(">>>>Token: " + t);
			double[][] prevcurr = new double[numLabels][numLabels];
			for (int s = 0; s < numLabels; s++) {
				//System.out.println("labelScores[" + s + "]" + labelScores[s]);
				computeVitLabelScores(t, s, sentence, prevcurr[s]);
				//System.out.println("prevcurr[" + s + "] " + priArr(prevcurr[s]));
				ArrayUtil.logNormalize(prevcurr[s]);
				prevcurr[s] = ArrayUtil.add(prevcurr[s], labelScores[s]);
			}
			//System.out.println("prevcurr: ");
			//u.p(prevcurr);
			for (int s = 0; s < numLabels; s++) {
				double[] sprobs = u.getColumn(prevcurr, s);
				if (t == divergePoint) {
					bptr[t][s] = u.nthLargest(2, sprobs);
				} else {
					bptr[t][s] = ArrayUtil.argmax(sprobs); // u.nthLargest(2, sprobs);
				}

				vit[t][s] = sprobs[bptr[t][s]];
			}
			labelScores = vit[t];
		}

		//System.out.print("vit[][] = ");
		//u.p(vit);
		
		// multiple paths produced via viterbi methods.
		int[][] viterbiPaths = new int[numLabels][T];
		double[] probs = new double[numLabels];
        double confs[][] = new double[numLabels][T];
		// for each row in the viterbi matrix (rows = labels : columns = tokens)
		for (int d = 0; d < vit[T - 1].length; d++) {

			// sentence.labels[T-1] = u.nthLargest(d, vit[T-1]); //ArrayUtil.argmax(vit[T-1]);
			viterbiPaths[d][T - 1] = u.nthLargest(d + 1, vit[T - 1]);
			//Util.p(vit[T - 1]);
			//System.out.println("` " + viterbiPaths[d][T - 1] + " asd "
			//		+ labelVocab.name(viterbiPaths[d][T - 1]));
			//System.out.print("***" + labelVocab.name(viterbiPaths[d][T - 1]));
			//System.out.println(" with prob: " + Math.exp(vit[T - 1][viterbiPaths[d][T - 1]]));
			double unNormalProb = vit[T - 1][viterbiPaths[d][T - 1]];
            confs[d][T-1] = vit[T - 1][viterbiPaths[d][T - 1]];

			int backtrace = bptr[T - 1][viterbiPaths[d][T - 1]];
			for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
				// sentence.labels[i] = backtrace;
				viterbiPaths[d][i] = backtrace;
				//System.out.println(labelVocab.name(backtrace) + " with prob: "
				//		+ Math.exp(vit[i][backtrace]));
				unNormalProb += vit[i][backtrace];
                confs[d][i] =Math.exp(vit[i][backtrace]); //vit[T - 1][viterbiPaths[d][i]];
				backtrace = bptr[i][backtrace];
			}

			assert (backtrace == m.startMarker());
			probs[d] = Math.exp(unNormalProb);
		}
		//sentence.labels = viterbiPaths[1];
		vp.addPaths(viterbiPaths);
		vp.addProbs(probs);
        vp.addConfs(confs);
	
		return vp;
	}
	
	//TODO remove
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

}
