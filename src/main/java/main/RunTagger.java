package main;

import impl.ModelSentence;
import impl.Sentence;
import impl.decoders.IDecoder;
import impl.decoders.greedy.Greedy;
import impl.decoders.viterbi.*;
import impl.features.WordClusterPaths;
import io.CoNLLReader;
import io.JsonTweetReader;

import java.io.*;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
import util.BasicFileIO;
import edu.stanford.nlp.util.StringUtils;

/**
 * Commandline interface to run the Twitter POS tagger with a variety of possible input and output formats.
 * Also does basic evaluation if given labeled input text.
 * 
 * For basic usage of the tagger from Java, see instead Tagger.java.
 */
public class RunTagger {
	Tagger tagger;
	
	// Commandline I/O-ish options
	String inputFormat = "conll"; //"auto";
	String outputFormat = "auto"; //"conll";//"auto";
	int inputField = 1;
	
	String inputFilename;
	String outputFilename;
	/** Can be either filename or resource name **/
	String modelFilename = "src/main/resources/model.20120919";

	public boolean noOutput = false;
	public boolean justTokenize = false;
	
	public static enum Decoder { GREEDY, VITERBI, VITERBIARRAY, VITERBINBEST, VITERBIEFF,VITERBIDIV};
	public Decoder decoder = Decoder.VITERBIEFF;
	public boolean showConfidence = true;

	PrintStream outputStream;
	Iterable<Sentence> inputIterable = null;
	
	// Evaluation stuff
	private static HashSet<String> _wordsInCluster;
	// Only for evaluation mode (conll inputs)
	int numTokensCorrect = 0;
	int numTokens = 0;
	int oovTokensCorrect = 0;
	int oovTokens = 0;
	int clusterTokensCorrect = 0;
	int clusterTokens = 0;


	public static void die(String message) {
		// (BTO) I like "assert false" but assertions are disabled by default in java
		System.err.println(message);
		System.exit(-1);
	}
	public RunTagger() throws UnsupportedEncodingException {
		// force UTF-8 here, so don't need -Dfile.encoding
		this.outputStream = new PrintStream(System.out, true, "UTF-8");
	}


    public RunTagger(String modelfile, String input, Decoder decoder) throws UnsupportedEncodingException {
        // force UTF-8 here, so don't need -Dfile.encoding
        this.outputStream = new PrintStream(System.out, true, "UTF-8");
        this.modelFilename = modelfile;
        this.inputFilename = input;
        this.decoder = decoder;


    }

	public void detectAndSetInputFormat(String tweetData) throws IOException {
		JsonTweetReader jsonTweetReader = new JsonTweetReader();
		if (jsonTweetReader.isJson(tweetData)) {
			System.err.println("Detected JSON input format");
			inputFormat = "json";
		} else {
			System.err.println("Detected text input format");
			inputFormat = "text";
		}
	}

	public void runTagger() throws IOException, ClassNotFoundException {
		
		System.out.println("In Tag mode!");
		
		tagger = new Tagger();
		if (!justTokenize) {
			tagger.loadModel(modelFilename);
            System.out.println(modelFilename);
        }
		
		if (inputFormat.equals("conll")) { //(true){//
			runTaggerInEvalMode();
			return;
		} 

		JsonTweetReader jsonTweetReader = new JsonTweetReader();
		
		LineNumberReader reader = new LineNumberReader(BasicFileIO.openFileToReadUTF8(inputFilename));
		String line;
		long currenttime = System.currentTimeMillis();
		int numtoks = 0;
		while ( (line = reader.readLine()) != null) {
			String[] parts = line.split("\t");
			String tweetData = parts[inputField-1];
			
			if (reader.getLineNumber()==1) {
				if (inputFormat.equals("auto")) {
					detectAndSetInputFormat(tweetData);
				}
			}
			
			String text;
			if (inputFormat.equals("json")) {
				text = jsonTweetReader.getText(tweetData);
				if (text==null) {
					System.err.println("Warning, null text (JSON parse error?), using blank string instead");
					text = "";
				}
			} else {
				text = tweetData;
			}
			
			Sentence sentence = new Sentence();
			sentence.tokens = Twokenize.tokenizeRawTweetText(text);
			ModelSentence modelSentence = null;

			if (sentence.T() > 0 && !justTokenize) {
				modelSentence = new ModelSentence(sentence.T());
				tagger.featureExtractor.computeFeatures(sentence, modelSentence);
				goDecode(modelSentence);
			}
				
			if (outputFormat.equals("conll")) {
				outputJustTagging(sentence, modelSentence);
			} else {
				
				//for(int t = 0; t<modelSentence.T; t++)
				//{
					//System.out.println("asdfasdfadsf:::: "+ tagger.model.labelVocab.name(modelSentence.labels[t]));
				//}
				
				//System.out.println("outputPrepend...");
				outputPrependedTagging(sentence, modelSentence, justTokenize, line);
			}
			numtoks += sentence.T();
		}
		long finishtime = System.currentTimeMillis();
		System.err.printf("Tokenized%s %d tweets (%d tokens) in %.1f seconds: %.1f tweets/sec, %.1f tokens/sec\n",
				justTokenize ? "" : " and tagged", 
				reader.getLineNumber(), numtoks, (finishtime-currenttime)/1000.0,
				reader.getLineNumber() / ((finishtime-currenttime)/1000.0),
				numtoks / ((finishtime-currenttime)/1000.0)
		);
		reader.close();
	}

	/** Runs the correct algorithm (make config option perhaps) **/
	public void goDecode(ModelSentence mSent) {
		if (decoder == Decoder.GREEDY) {
			//System.out.println("Running GREEDY decode()");
			//tagger.model.greedyDecode(mSent, showConfidence);
			Greedy greedy = new Greedy(tagger.model);
			greedy.decode(mSent);
		} else if (decoder == Decoder.VITERBI) {
			// if (showConfidence) throw new RuntimeException - < Don't do this anymore
			//System.out.println("Running VITERBI decode()");
			//tagger.model.viterbiDecode(mSent);
			IDecoder diverge = new Viterbi(tagger.model);
			diverge.decode(mSent);
		} //VITERBINBEST
		else if (decoder == Decoder.VITERBIARRAY) {
			//System.out.println("Running VITERBIARRAY decode()");
			//tagger.model.viterbiDecode(mSent);
			IDecoder diverge = new ViterbiArrayTree(tagger.model);
			diverge.decode(mSent);
		}
		else if (decoder == Decoder.VITERBINBEST) {
			//System.out.println("Running VITERBIARRAY decode()");
			//tagger.model.viterbiDecode(mSent);
			IDecoder diverge = new ViterbiNBest(tagger.model);
			diverge.decode(mSent);
		}
        else if (decoder == Decoder.VITERBIEFF) {
            //System.out.println("Running VITERBIARRAY decode()");
            //tagger.model.viterbiDecode(mSent);
            IDecoder diverge = new ViterbiTableEfficientThird(tagger.model);
            diverge.decode(mSent);
        }
        else if (decoder == Decoder.VITERBIDIV) {
            //System.out.println("Running VITERBIARRAY decode()");
            //tagger.model.viterbiDecode(mSent);
            IDecoder diverge = new ViterbiDiverge(tagger.model);
            diverge.decode(mSent);
        }
		
	}
	
	public void runTaggerInEvalMode() throws IOException, ClassNotFoundException {
		
		System.out.println("In Eval mode!");
		
		long t0 = System.currentTimeMillis();
		int n=0;

		List<Sentence> examples = CoNLLReader.readFile(inputFilename);
        inputIterable = examples;
        System.out.println("examples:"+examples.size());
        //int[][] confusion = new int[tagger.model.numLabels][tagger.model.numLabels];
		
		for (Sentence sentence : examples) {
			n++;
            System.out.println(sentence.toString());
            ModelSentence mSent = new ModelSentence(sentence.T());
			tagger.featureExtractor.computeFeatures(sentence, mSent);
			goDecode(mSent);
			
			if ( ! noOutput) {
				outputJustTagging(sentence, mSent);	
			}
			//evaluateSentenceTagging(sentence, mSent);
			//evaluateCompleteSentenceTagging(sentence,mSent);
			//evaluateSentenceTaggingNPath(sentence, mSent);
			evaluateCompleteSentenceTaggingNPath(sentence, mSent);
			//evaluateOOV(sentence, mSent);
			//getconfusion(sentence, mSent, confusion);
		}

		System.err.printf("%d / %d correct = %.4f acc, %.4f err\n", 
				numTokensCorrect, numTokens,
				numTokensCorrect*1.0 / numTokens,
				1 - (numTokensCorrect*1.0 / numTokens)
		);
		double elapsed = ((double) (System.currentTimeMillis() - t0)) / 1000.0;
		System.err.printf("%d tweets in %.1f seconds, %.1f tweets/sec\n",
				n, elapsed, n*1.0/elapsed);

        System.out.println(decoder.toString());

        File results = new File("/Volumes/LocalDataHD/ps324/exp1.txt");
        FileUtils.writeStringToFile(results,(decoder.toString()+"\n"),true);
        FileUtils.writeStringToFile(results,""+n+" tweets in "+ elapsed +" seconds, :"+(n*1.0/elapsed)+" tweets/sec\n",true);
        FileUtils.writeStringToFile(results,""+numTokensCorrect+"/"+numTokens+", correct, "+(numTokensCorrect*1.0 / numTokens)+
                                    " acc, "+(1 - (numTokensCorrect*1.0 / numTokens)) + " err\n",true);


/*		System.err.printf("%d / %d cluster words correct = %.4f acc, %.4f err\n", 
				oovTokensCorrect, oovTokens,
				oovTokensCorrect*1.0 / oovTokens,
				1 - (oovTokensCorrect*1.0 / oovTokens)
		);	*/
/*		int i=0;
		System.out.println("\t"+tagger.model.labelVocab.toString().replaceAll(" ", ", "));
		for (int[] row:confusion){
			System.out.println(tagger.model.labelVocab.name(i)+"\t"+Arrays.toString(row));
			i++;
		}		*/
	}
	
	private void evaluateOOV(Sentence lSent, ModelSentence mSent) throws FileNotFoundException, IOException, ClassNotFoundException {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			if(wordsInCluster().contains(lSent.tokens.get(t))){
				oovTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
				oovTokens += 1;
			}
		}
    }
	private void getconfusion(Sentence lSent, ModelSentence mSent, int[][] confusion) {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			if(trueLabel!=-1)
				confusion[trueLabel][predLabel]++;
		}
		
		
    }
	public void evaluateSentenceTagging(Sentence lSent, ModelSentence mSent) {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			numTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
			numTokens += 1;
		}
	}
	
	public void evaluateCompleteSentenceTagging(Sentence lSent, ModelSentence mSent) {
		int tempNumTokensCorrect = 0;
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			tempNumTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
			
		}
		if(tempNumTokensCorrect == mSent.T)
		{
			numTokensCorrect +=1;
		}
		numTokens += 1;
	}
	
	public void evaluateCompleteSentenceTaggingNPath(Sentence lSent, ModelSentence mSent) {
		int tempNumTokensCorrect = 0;
		int maxCorrect = 0;
		for(int i=0; i<mSent.nPaths.length; i++)
		{
			tempNumTokensCorrect = 0;
			for (int t=0; t < mSent.T; t++) {
				int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
				int predLabel = mSent.nPaths[i][t];
				tempNumTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
			}
			if (tempNumTokensCorrect > maxCorrect)
			{
				maxCorrect = tempNumTokensCorrect;
			}
		}
		if(maxCorrect == mSent.T)
		{
			numTokensCorrect += 1;
		}
		numTokens += 1;
	}
	
	public void evaluateSentenceTaggingNPath(Sentence lSent, ModelSentence mSent) {
		int tempNumTokensCorrect = 0;
		int maxCorrect = 0;
		for(int i=0; i<mSent.nPaths.length; i++)
		{
			tempNumTokensCorrect = 0;
			for (int t=0; t < mSent.T; t++) {
				int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
				int predLabel = mSent.nPaths[i][t];
				tempNumTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
			}
			if (tempNumTokensCorrect > maxCorrect)
			{
				maxCorrect = tempNumTokensCorrect;
			}
		}
		numTokensCorrect+=maxCorrect;
		numTokens +=mSent.T;
		
		
	}
	
	private String formatConfidence(double confidence) {
		// too many decimal places wastes space
		return String.format("%.4f", confidence);
	}

	/**
	 * assume mSent's labels hold the tagging.
	 */
	public void outputJustTagging(Sentence lSent, ModelSentence mSent) {
		// mSent might be null!

		if (outputFormat.equals("conll")) {
			for (int t=0; t < lSent.T(); t++) {
				outputStream.printf("%s\t%s", 
						lSent.tokens.get(t),  
						tagger.model.labelVocab.name(mSent.labels[t]));
				if (mSent.confidences != null) {
					outputStream.printf("\t%s", formatConfidence(mSent.confidences[0][t]));
				}
				outputStream.printf("\n");
			}
			outputStream.println("");
		} 
		else {
			die("bad output format for just tagging: " + outputFormat);
		}
	}
	/**
	 * assume mSent's labels hold the tagging.
	 * 
	 * @param lSent
	 * @param mSent
	 * @param inputLine -- assume does NOT have trailing newline.  (default from java's readLine)
	 */
	public void outputPrependedTagging(Sentence lSent, ModelSentence mSent, 
			boolean suppressTags, String inputLine) {
		// mSent might be null!
		for (int k = 0; k<mSent.K; k++) {
            int T = lSent.T();
            String[] tokens = new String[T];
            String[] tags = new String[T];
            String[] confs = new String[T];
            for (int t = 0; t < T; t++) {
                tokens[t] = lSent.tokens.get(t);
                if (!suppressTags) {
                    tags[t] = tagger.model.labelVocab.name(mSent.nPaths[k][t]);
                }
                if (showConfidence) {
                    confs[t] = formatConfidence(mSent.confidences[k][t]);
                }
            }

            StringBuilder sb = new StringBuilder();
            sb.append(StringUtils.join(tokens));
            sb.append("\t");
            if (!suppressTags) {
                sb.append(StringUtils.join(tags));
                sb.append("\t");
            }
            if (showConfidence) {
                sb.append(StringUtils.join(confs));
                sb.append("\t");
                sb.append(mSent.pathConfidences[k]);
                sb.append("\t");
                sb.append("\t");
            }
            sb.append(inputLine);

            outputStream.println(sb.toString());
        }
	}


	///////////////////////////////////////////////////////////////////


	public static void main(String[] args) throws IOException, ClassNotFoundException {        
		if (args.length > 0 && (args[0].equals("-h") || args[0].equals("--help"))) {
			usage();
		}

		RunTagger tagger = new RunTagger();

		int i = 0;
		while (i < args.length) {
			if (!args[i].startsWith("-")) {
				break;
			} else if (args[i].equals("--model")) {
				tagger.modelFilename = args[i+1];
				i += 2;
			} else if (args[i].equals("--just-tokenize")) {
				tagger.justTokenize = true;
				i += 1;
			} else if (args[i].equals("--decoder")) {
				if (args[i+1].equals("viterbi")) tagger.decoder = Decoder.VITERBI;
				else if (args[i+1].equals("greedy"))  tagger.decoder = Decoder.GREEDY;
				else die("unknown decoder " + args[i+1]);
				i += 2;
			} else if (args[i].equals("--quiet")) {
				tagger.noOutput = true;
				i += 1;
			} else if (args[i].equals("--input-format")) {
				String s = args[i+1];
				if (!(s.equals("json") || s.equals("text") || s.equals("conll")))
					usage("input format must be: json, text, or conll");
				tagger.inputFormat = args[i+1];
				i += 2;
			} else if (args[i].equals("--output-format")) {
				tagger.outputFormat = args[i+1];
				i += 2;
			} else if (args[i].equals("--output-file")) {
				tagger.outputFilename = args[i+1];
				tagger.outputStream = new PrintStream(new FileOutputStream(tagger.outputFilename, true));
				i += 2;
			} else if (args[i].equals("--output-overwrite")) {
				tagger.outputStream = new PrintStream(new FileOutputStream(tagger.outputFilename, false));
				i += 1;
			} else if (args[i].equals("--input-field")) {
				tagger.inputField = Integer.parseInt(args[i+1]);
				i += 2;
			} else if (args[i].equals("--word-clusters")) {
				WordClusterPaths.clusterResourceName = args[i+1];
				i += 1;
			} else if (args[i].equals("--no-confidence")) {
				tagger.showConfidence = false;
				i += 1;
			}	
			else {
				System.out.println("bad option " + args[i]);
				usage();                
			}
		}
		
		if (args.length - i > 1) usage();
		if (args.length == i || args[i].equals("-")) {
			System.err.println("Listening on stdin for input.  (-h for help)");
			//tagger.inputFilename = new Scanner(System.in).nextLine(); // TODO replace for changable file name on running
            tagger.inputFilename = "/Volumes/LocalDataHD/ps324/tweets";
		} else {
			tagger.inputFilename = args[i];
		}
		
		tagger.finalizeOptions();
		
		tagger.runTagger();		
	}
	
	public void finalizeOptions() throws IOException {
		if (outputFormat.equals("auto")) {
			if (inputFormat.equals("conll")) {
				outputFormat = "conll";
			} else {
				outputFormat = "pretsv";
			}
		}
		if (showConfidence && decoder==Decoder.VITERBI) {
			System.err.println("Confidence output is unimplemented in Viterbi, turning it off.");
			showConfidence = false;
		}
		if (justTokenize) {
			showConfidence = false;
		}
	}
	
	public static void usage() {
		usage(null);
	}

	public static void usage(String extra) {
		System.out.println(
"RunTagger [options] [ExamplesFilename]" +
"\n  runs the CMU ARK Twitter tagger on tweets from ExamplesFilename, " +
"\n  writing taggings to standard output. Listens on stdin if no input filename." +
"\n\nOptions:" +
"\n  --model <Filename>        Specify model filename. (Else use built-in.)" +
"\n  --just-tokenize           Only run the tokenizer; no POS tags." +
"\n  --quiet                   Quiet: no output" +
"\n  --input-format <Format>   Default: auto" +
"\n                            Options: json, text, conll" +
"\n  --output-format <Format>  Default: automatically decide from input format." +
"\n                            Options: pretsv, conll" +
"\n  --output-file <Filename>  Save output to specified file (Else output to stdout)" +
"\n  --output-overwrite        Overwrite output-file (default: append)" +
"\n  --input-field NUM         Default: 1" +
"\n                            Which tab-separated field contains the input" +
"\n                            (1-indexed, like unix 'cut')" +
"\n                            Only for {json, text} input formats." +
"\n  --word-clusters <File>    Alternate word clusters file (see FeatureExtractor)" +
"\n  --no-confidence           Don't output confidence probabilities" +
"\n  --decoder <Decoder>       Change the decoding algorithm (default: greedy)" +
"\n" +
"\nTweet-per-line input formats:" +
"\n   json: Every input line has a JSON object containing the tweet," +
"\n         as per the Streaming API. (The 'text' field is used.)" +
"\n   text: Every input line has the text for one tweet." +
"\nWe actually assume input lines are TSV and the tweet data is one field."+
"\n(Therefore tab characters are not allowed in tweets." +
"\nTwitter's own JSON formats guarantee this;" +
"\nif you extract the text yourself, you must remove tabs and newlines.)" +
"\nTweet-per-line output format is" +
"\n   pretsv: Prepend the tokenization and tagging as new TSV fields, " +
"\n           so the output includes a complete copy of the input." +
"\nBy default, three TSV fields are prepended:" +
"\n   Tokenization \\t POSTags \\t Confidences \\t (original data...)" +
"\nThe tokenization and tags are parallel space-separated lists." +
"\nThe 'conll' format is token-per-line, blank spaces separating tweets."+
"\n");
		
		if (extra != null) {
			System.out.println("ERROR: " + extra);
		}
		System.exit(1);
	}
	public static HashSet<String> wordsInCluster() {
		if (_wordsInCluster==null) {
			_wordsInCluster = new HashSet<String>(WordClusterPaths.wordToPath.keySet());
		}
		return _wordsInCluster;
	}
}
