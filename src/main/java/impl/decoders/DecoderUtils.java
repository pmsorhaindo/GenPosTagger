package impl.decoders;

import impl.Model;

public class DecoderUtils {
	
	private int numLabels;
	private Model m;
	
	public DecoderUtils(Model m){
		
	}
	
	/** Adds into labelScores **/
	public void computeBiasScores(double[] labelScores) {
		for (int k = 0; k < numLabels; k++) {
			labelScores[k] += m.biasCoefs[k];
		}
	}


}
