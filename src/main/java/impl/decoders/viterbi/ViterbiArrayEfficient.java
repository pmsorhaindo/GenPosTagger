package impl.decoders.viterbi;

import edu.berkeley.nlp.util.ArrayUtil;
import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;
import util.Tree;
import util.Util;

import java.util.ArrayList;
import java.util.Arrays;

public class ViterbiArrayEfficient implements IDecoder {

	private Model m;
	private int numLabels;
	private int N;
	private Util u;

	public ViterbiArrayEfficient(Model m){
		
		this.N = 2;
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
        double[][] labelScores = new double[N][numLabels];

        computeVitLabelScores(0, m.startMarker(), sentence, labelScores[0]);
        ArrayUtil.logNormalize(labelScores[0]);
//        computeVitLabelScores(0, m.startMarker(), sentence, labelScores[1]);
//        ArrayUtil.logNormalize(labelScores[1]);

        int[][] origPointer = new int[N][numLabels];
        for (int k = 0; k < numLabels; k++) {
            origPointer[0][k] = m.startMarker();
        }

        ArrayList<WordData> tokens = new ArrayList<>();
        tokens.add(new WordData());
        tokens.get(0).setData(labelScores);
        int[] dOffset = new int[numLabels];
        Arrays.fill(dOffset, 0);
        tokens.get(0).setDataOffset(dOffset);
        tokens.get(0).setPointer(origPointer);
        int[] pOffset = new int[numLabels];
        Arrays.fill(pOffset, 0);
        tokens.get(0).setPointerOffset(pOffset);

        // First word

        for(int a =1; a<T;a++) {
        double[][] prevcurr = new double[numLabels][numLabels];
        for (int s = 0; s < numLabels; s++) {
            //System.out.println("labelScores[" + s + "]" + labelScores[0][s]);
            computeVitLabelScores(a, s, sentence, prevcurr[s]);
            //System.out.println("prevcurr[" + s + "] " + ArrayUtil.toString(prevcurr[s]));
            ArrayUtil.logNormalize(prevcurr[s]);
            prevcurr[s] = ArrayUtil.add(prevcurr[s], tokens.get(a-1).getData()[0]);
        }
        System.out.println("prevCurr 1");
        //u.p(prevcurr);

        int bptr[][] = new int[N][numLabels];
        double vit[][] = new double[N][numLabels];
        for (int s = 0; s < numLabels; s++) {
            double[] sprobs = u.getColumn(prevcurr, s);
            bptr[0][s] = ArrayUtil.argmax(sprobs);
            vit[0][s] = sprobs[bptr[0][s]];
            //u.p(sprobs);
        }


        int[] offSetCounter = new int[numLabels];
        Arrays.fill(offSetCounter, 0);
        for (int i = 1; i < N; i++) {
            int[] offsetStore = new int[numLabels];
            offsetStore = tokens.get(0).getDataOffset();
            for (int s = 0; s < numLabels; s++) {
                double[] sprobs = u.getColumn(prevcurr, s);

                if (offsetStore[s] > 0) { //tokens.get(0).getDataOffset()[ArrayUtil.argmax(sprobs)]
                    bptr[i][s] = ArrayUtil.argmax(sprobs); // TODO check bptr assignments is ok might need to ref which offset! (maybe remove the for loop and make a = 1 and a-1 = 0 )
                    vit[i][s] = sprobs[bptr[i][s]];
                    offsetStore[s] = offsetStore[s] - 1;
                } else {
                    //int nth = offSetCounter[s]-N;
                    bptr[i][s] = u.nthLargest(2, sprobs); // i+1 of some nth thing above ????
                    vit[i][s] = sprobs[bptr[i][s]];
                }
            }

        }

        tokens.add(new WordData());
        tokens.get(a).setData(vit);
        tokens.get(a).setPointer(bptr);
    }

        tokens.get(0).print();
        tokens.get(1).print();
        tokens.get(2).print();
        tokens.get(3).print();
        tokens.get(4).print();

        System.out.println("suuup!?!");

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

    public class WordData {

        double[][] data = new double[N][numLabels];
        int[] dataOffset = new int[N];
        int[][] pointer = new int[N][numLabels];
        int[] pointerOffset = new int[N];

        public WordData(){

        }

        public double[][] getData() {
            return data;
        }

        public void setData(double[][] data) {
            this.data = data;
        }

        public int[] getDataOffset() {
            return dataOffset;
        }

        public void setDataOffset(int[] dataOffset) {
            this.dataOffset = dataOffset;
        }

        public int[][] getPointer() {
            return pointer;
        }

        public void setPointer(int[][] pointer) {
            this.pointer = pointer;
        }

        public int[] getPointerOffset() {
            return pointerOffset;
        }

        public void setPointerOffset(int[] pointerOffset) {
            this.pointerOffset = pointerOffset;
        }


        public void print(){
            u.p(this.data);
        }

    }
}