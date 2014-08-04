package impl.decoders.viterbi;

import edu.berkeley.nlp.util.ArrayUtil;
import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;
import util.Util;

import java.util.ArrayList;
import java.util.Arrays;

public class ViterbiArrayEfficient implements IDecoder {

    private Model m;
    private int numLabels;
    private int N;
    private Util u;

    public ViterbiArrayEfficient(Model m) {

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
        int[][] origPointer = new int[N][numLabels];
        int[][] pOffset = new int[N][numLabels];
        for (int i = 0; i < N; i++) {
            computeVitLabelScores(0, m.startMarker(), sentence, labelScores[i]);
            ArrayUtil.logNormalize(labelScores[i]);
            for(int j=0;j<numLabels;j++) {
                origPointer[i][j] = m.startMarker(); // TODO check if this and the following are needed. and rename to make it obvious
                pOffset[i][j] = 0;
            }
        }

        ArrayList<WordData> tokens = new ArrayList<>();
        tokens.add(new WordData());
        tokens.get(0).setData(labelScores);
        int[] dOffset = new int[numLabels];
        Arrays.fill(dOffset, 0);
        tokens.get(0).setDataOffset(dOffset);
        tokens.get(0).setPointer(origPointer);
        Arrays.fill(pOffset[0], 0);
        tokens.get(0).setPointerOffset(pOffset);


        for (int h = 1; h < T; h++) {
            tokens.add(new WordData());

            double[][] prevcurr = new double[numLabels][numLabels];
            int bptr[][] = new int[N][numLabels];
            double vit[][] = new double[N][numLabels];
            int[][] offsetStore = new int[N][numLabels];

            for (int s = 0; s < numLabels; s++) {
                //System.out.println("labelScores[" + s + "]" + labelScores[0][s]);
                computeVitLabelScores(h, s, sentence, prevcurr[s]);
                //System.out.println("prevcurr[" + s + "] " + ArrayUtil.toString(prevcurr[s]));
                ArrayUtil.logNormalize(prevcurr[s]);
                prevcurr[s] = ArrayUtil.add(prevcurr[s], tokens.get(h-1).getData()[0][s]);
            }

            for (int i = 0; i < N; i++) {
                int[] labelOffSetCounter = new int[numLabels];
                for (int s = 0; s < numLabels; s++) {
                    double[] sprobs = u.getColumn(prevcurr, s);
                    if (i == 0) {
                        bptr[i][s] = ArrayUtil.argmax(sprobs);
                        vit[i][s] = sprobs[bptr[i][s]];
                        labelOffSetCounter[s] = i;
                    }
                    else { //if(tokens.get(h-1).getPointerOffset()[i][s]>i) {
                        //System.out.println("swap value out");
                        computeVitLabelScores(h, s, sentence, prevcurr[s]);
                        ArrayUtil.logNormalize(prevcurr[s]);
                        //here be dragons!
                        double[] prevcurrline = ArrayUtil.add(prevcurr[s], tokens.get(h-1).getData()[i][s]);
                        //System.out.println("***");
                        //u.p(sprobs);
                        double[] sprobsTemp = sprobs.clone();
                        //u.p(sprobsTemp);
                        //u.p(sprobs);
                        sprobsTemp[ArrayUtil.argmax(sprobs)] = prevcurrline[ArrayUtil.argmax(sprobs)];
                        //u.p(sprobsTemp);
                        //u.p(sprobs);
                        //System.out.println("out: " + ArrayUtil.argmax(sprobs));
                        //System.out.println("in: " + ArrayUtil.argmax(sprobsTemp));
                        bptr[i][s] = ArrayUtil.argmax(sprobsTemp);
                        vit[i][s] = sprobsTemp[bptr[i][s]];

                        int x = ArrayUtil.argmax(sprobs);

                        if(bptr[i][s] == ArrayUtil.argmax(sprobs)) {
                            System.out.println("point "+( (int) labelOffSetCounter[s] + 1));
                            labelOffSetCounter[s] = labelOffSetCounter[s] + 1;
                        }
                    }
//                    else {
//                        System.out.println(s+" special!");
//                        bptr[i][s] = u.nthLargest(i+1, sprobs); // i+1 of some nth thing above ????
//                        vit[i][s] = sprobs[bptr[i][s]];
//                        labelOffSetCounter[s] = i+1;
//                    }
                }
                offsetStore[i] = labelOffSetCounter;
            }
            tokens.get(h).setData(vit);
            tokens.get(h).setPointer(bptr);
            tokens.get(h).setPointerOffset(offsetStore);
        }

        tokens.get(0).print();
        tokens.get(1).print();
        tokens.get(2).print();
        tokens.get(3).print();
        tokens.get(4).print();

        //[14, 6, 2, 4, 1, 7]

        sentence.labels[T - 1] = ArrayUtil.argmax(tokens.get(T-1).getData()[1]);
        System.out.println("***" + m.labelVocab.name(sentence.labels[T - 1]));
//        double prob = vit[T - 1][sentence.labels[T - 1]]; //Math.exp(vit[T - 1][sentence.labels[T - 1]]);
//        System.out.println(" with prob: " + prob);
//        this.probs.add(prob);

        int backtrace = tokens.get(T-1).getPointer()[tokens.get(T-1).getPointerOffset()[1][sentence.labels[T - 1]]][sentence.labels[T - 1]];
        for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
            sentence.labels[i] = backtrace;
//            double newProb = vit[i][backtrace]; //Math.exp(vit[i][backtrace]);
            System.out.println("***" + m.labelVocab.name(backtrace));
//                    + " with prob: " + newProb);
//            this.probs.add(newProb);
            backtrace = tokens.get(i).getPointer()[tokens.get(T-1).getPointerOffset()[1][backtrace]][backtrace];
        }

        System.out.println();
        sentence.labels[T - 1] = ArrayUtil.argmax(tokens.get(T-1).getData()[0]);
        System.out.println("***" + m.labelVocab.name(sentence.labels[T - 1]));
        backtrace = tokens.get(T-1).getPointer()[tokens.get(T-1).getPointerOffset()[1][sentence.labels[T - 1]]][sentence.labels[T - 1]];
        for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
            sentence.labels[i] = backtrace;
//            double newProb = vit[i][backtrace]; //Math.exp(vit[i][backtrace]);
            System.out.println("***" + m.labelVocab.name(backtrace));
//                    + " with prob: " + newProb);
//            this.probs.add(newProb);
            backtrace = tokens.get(i).getPointer()[tokens.get(T-1).getPointerOffset()[0][backtrace]][backtrace];
        }
        assert (backtrace == m.startMarker());
    }

    // TODO remove
    public void computeVitLabelScores(int t, int prior, ModelSentence sentence,
                                      double[] labelScores) {
        Arrays.fill(labelScores, 0);
        m.computeBiasScores(labelScores);
        viterbiEdgeScores(prior, sentence, labelScores);
        m.computeObservedFeatureScores(t, sentence, labelScores);
    }

    /**
     * @return dim T array s.t. labelScores[t]+=score of label prior followed by
     * label t
     */
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
        int[][] pointerOffset = new int[N][numLabels];
        int[] pointerOffsetOld = new int[N];

        public WordData() {

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

        public int[] getPointerOffsetOld() {
            return pointerOffsetOld;
        }

        public void setPointerOffsetOld(int[] pointerOffsetOld) {
            this.pointerOffsetOld = pointerOffsetOld;
        }

        public int[][] getPointerOffset() {
            return pointerOffset;
        }

        public void setPointerOffset(int[][] pointerOffset) {
            this.pointerOffset = pointerOffset;
        }

        public void print() {
            u.p(this.data);
            u.p(this.pointer);
            u.p(this.pointerOffset);
        }

    }
}