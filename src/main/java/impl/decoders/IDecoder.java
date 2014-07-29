package impl.decoders;

import impl.ModelSentence;


public interface IDecoder {
	
	// main functionality leaves ModelSentence updated with labels.
	void decode(ModelSentence sentence);
	
	// return the settings for the viterbi decoder.
	String decodeSettings();
	
}
