package impl.features;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;

import main.Twokenize;
import impl.features.FeatureExtractor.FeatureExtractorInterface;
import impl.features.FeatureExtractor.PositionFeaturePairs;
import util.BasicFileIO;
import edu.stanford.nlp.util.StringUtils;

/**
 * Brown word clusters: features are path prefixes down the tree. 
 **/
public class WordClusterPaths implements FeatureExtractorInterface {

	/** TODO this should be moved into config somehow **/
	public static String clusterResourceName = "/50mpaths2";

	public static HashMap<String,String> wordToPath;

	public WordClusterPaths() throws IOException {
//		log.info("Loading clusters");

		//read in paths file
		BufferedReader bReader = BasicFileIO.getResourceReader(clusterResourceName);
		String[] splitline = new String[3];
		String line=BasicFileIO.getLine(bReader);
		wordToPath = new HashMap<String,String>(); 
		while(line != null){
			splitline = line.split("\\t");
			wordToPath.put(splitline[1], splitline[0]);
			line = BasicFileIO.getLine(bReader);
		}			
//		log.info("Finished loading clusters");
	}

	public void addFeatures(List<String> tokens, PositionFeaturePairs pairs) {
		
		System.out.println("tokens: " + tokens.toString());
		
		String bitstring = null;
		for (int t=0; t < tokens.size(); t++) {
			String tok = tokens.get(t);
		    String normaltok = FeatureUtil.normalize(tok);
		    
		    //System.out.println("forToken: " + normaltok);
		    
		    bitstring = wordToPath.get(normaltok);
		    
		    if (bitstring == null){
		    	
		    	List fuzzes = FeatureUtil.fuzztoken(normaltok, true);
		    	//System.out.println("Fuzzes: "+ fuzzes.toString());
		    	
		    	for (String fuzz : FeatureUtil.fuzztoken(normaltok, true)){
		    		bitstring = wordToPath.get(fuzz);
		    		if (bitstring != null){
		    			//System.err.println(normaltok+"->"+fuzz);
		    			break;
		    		}
		    	}
		    }

			if (bitstring != null){
				int i;
				bitstring = StringUtils.pad(bitstring, 16).replace(' ', '0');
				//System.out.println("Padded bitstring: "+ bitstring);
				for(i=2; i<=16; i+=2){
					pairs.add(t, "BigCluster|" + bitstring.substring(0,i));
					//System.out.println("adding big cluster pair: "+ bitstring.substring(0,i));
				}
				if (t<tokens.size()-1){
					for(i=4; i<=12; i+=4)
						pairs.add(t+1, "PrevBigCluster"+"|" + bitstring.substring(0,i));
						//System.out.println("adding prev big pair: "+ bitstring.substring(0,i));
				}
				if (t>0){
					for(i=4; i<=12; i+=4)
						pairs.add(t-1, "NextBigCluster"+"|" + bitstring.substring(0,i));
						//System.out.println("adding next big pair: "+ bitstring.substring(0,i));
				}
			}
/*				else{
				pairs.add(t, "BigCluster|none");
			}*/
		}
		
		//System.out.println("pairs: "+ pairs.toString());
	}
}