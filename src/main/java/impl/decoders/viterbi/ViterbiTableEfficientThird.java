package impl.decoders.viterbi;

import edu.berkeley.nlp.util.ArrayUtil;
import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;
import util.Util;

import java.util.ArrayList;
import java.util.Arrays;

public class ViterbiTableEfficientThird implements IDecoder {

    private Model m;
    private int numLabels;
    private int K;
    private Util u;

    public ViterbiTableEfficientThird(Model m) {

        this.K = 3;
        this.u = new Util();
        this.m = m;
        this.numLabels = m.labelVocab.size();
        assert numLabels != 0;

    }

    @Override
    public void decode(ModelSentence sentence) {
        viterbiTableDecode(sentence);

    }

    @Override
    public String decodeSettings() {
        // TODO Auto-generated method stub
        return null;
    }


    public void viterbiTableDecode(ModelSentence sentence) {

        ArrayList<WordData> tokens = new ArrayList<>();

        int T = sentence.T; // Number of tokens to be tagged.
        double[][] labelScores = new double[K][numLabels];
        int[][] origTagPointer = new int[K][numLabels];
        int[][] origTagVPointer = new int[K][numLabels];
        for (int i = 0; i < K; i++) {
            computeVitLabelScores(0, m.startMarker(), sentence, labelScores[i]);
            ArrayUtil.logNormalize(labelScores[i]);
            for(int s=0;s<numLabels;s++) {
                origTagPointer[i][s] = m.startMarker();
                origTagVPointer[i][s] = 0;
            }
        }

        tokens.add(new WordData());
        tokens.get(0).setData(labelScores);
        tokens.get(0).setTagPointer(origTagPointer);
        tokens.get(0).setTagVersionPointer(origTagVPointer);

        for (int i = 1; i < T; i++) {
            tokens.add(new WordData());

            double[][] prevcurr = new double[numLabels][numLabels];
            double[][] vit = new double[K][numLabels];
            int[][] tagPointers = new int[K][numLabels];
            int[][] tagVPointers = new int[K][numLabels];
            double[][] sprobs = new double[numLabels][numLabels];

            for (int s = 0; s < numLabels; s++) {
                computeVitLabelScores(i, s, sentence, prevcurr[s]);
                ArrayUtil.logNormalize(prevcurr[s]);
                prevcurr[s] = ArrayUtil.add(prevcurr[s], tokens.get(i-1).getData()[0][s]);
            }

            for(int s = 0; s<numLabels; s++)
            {
                sprobs[s] = u.getColumn(prevcurr, s);
            }

            int[][] labelUsageCounter = new int[numLabels][numLabels];
            for (int j = 0; j < K; j++) {
                for (int s = 0; s < numLabels; s++) {
                    //int maxTag = ArrayUtil.argmax(sprobs[s]);

                    /*if(labelUsageCounter[0][maxTag] < K)
                    {
                        int usage = labelUsageCounter[0][maxTag];

                        if(usage == 0)
                        {
                            tagPointers[j][s] = maxTag; // u.nthLargest(2, sprobs);
                            vit[j][s] = sprobs[s][tagPointers[j][s]];
                            tagVPointers[j][s] = usage;
                            labelUsageCounter[0][maxTag]++;
                        }
                        else // meaning -> if (usage < K)
                        {
                            double[] prevCurrLine = ArrayUtil.add(prevcurr[s], tokens.get(i - 1).getData()[usage][maxTag]);

                            sprobs[s][maxTag] = prevCurrLine[maxTag];

                            tagPointers[j][s] = ArrayUtil.argmax(sprobs[s]);
                            tagVPointers[j][s] = labelUsageCounter[0][ArrayUtil.argmax(sprobs[s])];
                            vit[j][s] = sprobs[s][tagPointers[j][s]];
                            labelUsageCounter[0][ArrayUtil.argmax(sprobs[s])]++;
                        }

                    }
                    else*/
                    {
                        boolean found = false;
                        int nextHighestFreeTag = 1;
                        while (!found) {
                            int maxTag = u.nthLargest(nextHighestFreeTag, sprobs[s]);
                            if (maxTag != -1 && labelUsageCounter[nextHighestFreeTag-1][maxTag] < K) {

                                int  usage = labelUsageCounter[nextHighestFreeTag-1][maxTag];
                                double[] prevCurrLine = ArrayUtil.add(prevcurr[s], tokens.get(i - 1).getData()[usage][maxTag]);

                                //if(usage != 0) sprobs[s][maxTag] = prevCurrLine[maxTag];

                                if(labelUsageCounter[nextHighestFreeTag-1][ArrayUtil.argmax(sprobs[s])]>=K)
                                {
                                    nextHighestFreeTag++;
                                    continue;
                                }

                                tagPointers[j][s] = ArrayUtil.argmax(sprobs[s]);
                                tagVPointers[j][s] = labelUsageCounter[nextHighestFreeTag-1][ArrayUtil.argmax(sprobs[s])];

                                vit[j][s] = sprobs[s][tagPointers[j][s]];
                                labelUsageCounter[nextHighestFreeTag-1][ArrayUtil.argmax(sprobs[s])]++;

                                found = true;
                            } else if (nextHighestFreeTag == numLabels) {
                                System.err.println("Ran out of Tags");
                                break;
                            } else {
                                nextHighestFreeTag++;
                            }
                        }
                    }
                }
            }
            //u.p(labelUsageCounter);
            u.p(tagPointers);
            //u.p(tagVPointers);
            tokens.get(i).setData(vit);
            tokens.get(i).setTagPointer(tagPointers);
            tokens.get(i).setTagVersionPointer(tagVPointers);
        }

        int k =0;

        sentence.labels[T - 1] = ArrayUtil.argmax(tokens.get(T-1).getData()[k]);
        System.out.println("***" + m.labelVocab.name(sentence.labels[T - 1]));
        int vPointer = tokens.get(T-1).getTagVersionPointer()[k][sentence.labels[T - 1]];
        int backtrace = tokens.get(T-1).getTagPointer() [vPointer] [sentence.labels[T - 1]];

        for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
            sentence.labels[i] = backtrace;
            System.out.println("***" + m.labelVocab.name(backtrace));
            vPointer = tokens.get(i).getTagVersionPointer()[vPointer][backtrace];
            backtrace = tokens.get(i).getTagPointer() [vPointer] [backtrace];
        }

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

        double[][] data = new double[K][numLabels];
        int[][] tagPointer = new int[K][numLabels];
        int[][] tagVersionPointer = new int[K][numLabels];
        double[][][]sprobs = new double[K][numLabels][numLabels];
        //TODO go back into sprobs and grab that DATA -> see Arrow on paper!!

        public WordData() {

        }

        public double[][] getData() {
            return data;
        }

        public void setData(double[][] data) {
            this.data = data;
        }

        public int[][] getTagPointer() {
            return tagPointer;
        }

        public void setTagPointer(int[][] tagPointer) {
            this.tagPointer = tagPointer;
        }

        public int[][] getTagVersionPointer() {
            return tagVersionPointer;
        }

        public void setTagVersionPointer(int[][] tagVersionPointer) {
            this.tagVersionPointer = tagVersionPointer;
        }

        public void print() {
            u.p(this.data);
            u.p(this.tagPointer);
            u.p(this.tagVersionPointer);
        }

    }
}