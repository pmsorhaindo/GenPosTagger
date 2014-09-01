package impl.decoders.greedy;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.DecoderUtils;
import impl.decoders.IDecoder;

import java.util.Arrays;

import util.Util;
import edu.stanford.nlp.math.ArrayMath;

public class Greedy  implements IDecoder {
	
	private Model m;
	private Util u;
	private DecoderUtils dUtils;
	private int numLabels;
    private final int K = 1;
	
	public Greedy(Model m) {
		
		this.numLabels = m.labelVocab.size();
		assert this.numLabels != 0;
		this.m = m;
		this.dUtils = new DecoderUtils(m);
		
	}

	@Override
	public void decode(ModelSentence sentence) {
        sentence.K = K;
		greedyDecode(sentence,false); //TODO marshall params for decode.
		
	}

	@Override
	public String decodeSettings() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * THIS CLOBBERS THE LABELS, stores its decoding into them. Does progressive
	 * rolling edge feature extraction
	 **/
	public void greedyDecode(ModelSentence sentence, boolean storeConfidences) {
		//System.out.println("Running the greedy decode.");

		int T = sentence.T;
		sentence.labels = new int[T];
		sentence.edgeFeatures[0] = m.startMarker();

		//System.out.println("sentence edge feature 1 aka startMarker(): "
		//		+ sentence.edgeFeatures[0]);
		//System.out.println("numLabels: " + numLabels + " T: " + T);

		if (storeConfidences)
			//TODO will need an extra dimension for multi-path greedy.
			sentence.confidences = new double[this.K][T];

		double[] labelScores = new double[numLabels];
		for (int t = 0; t < T; t++) {
			m.computeLabelScores(t, sentence, labelScores);
			sentence.labels[t] = ArrayMath.argmax(labelScores);
			if (t < T - 1)
				sentence.edgeFeatures[t + 1] = sentence.labels[t];
			if (storeConfidences) {
				ArrayMath.expInPlace(labelScores);
				double Z = ArrayMath.sum(labelScores);
				ArrayMath.multiplyInPlace(labelScores, 1.0 / Z);
				sentence.confidences[this.K][t] = labelScores[sentence.labels[t]];
			}
		}

        int[][] npaths = new int[K][sentence.T];
        npaths[0] = sentence.labels;
        sentence.nPaths = npaths;
	}

	/** Adds into labelScores **/
	public void computeEdgeScores(int t, ModelSentence sentence,
			double[] labelScores) {
		int prev = sentence.edgeFeatures[t];
		for (int k = 0; k < numLabels; k++) {
			labelScores[k] += m.edgeCoefs[prev][k];
		}
	}
}
