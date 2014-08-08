package impl.decoders.viterbi;

import edu.berkeley.nlp.util.ArrayUtil;
import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;
import util.Util;

import java.util.ArrayList;
import java.util.Arrays;

public class ViterbiTableEfficient implements IDecoder {

    private Model m;
    private int numLabels;
    private int N;
    private Util u;

    public ViterbiTableEfficient(Model m) {

        this.N = 3;
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
        double[][] labelScores = new double[N][numLabels];
        int[][] origTagPointer = new int[N][numLabels];
        int[][] origTagVPointer = new int[N][numLabels];
        for (int i = 0; i < N; i++) {
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
            double[][] vit = new double[N][numLabels];
            int[][] tagPointers = new int[N][numLabels];
            int[][] tagVPointers = new int[N][numLabels];

            for (int s = 0; s < numLabels; s++) {
                computeVitLabelScores(i, s, sentence, prevcurr[s]);
                ArrayUtil.logNormalize(prevcurr[s]);
                prevcurr[s] = ArrayUtil.add(prevcurr[s], tokens.get(i-1).getData()[0][s]);
            }

            int[] labelUsageCounter = new int[numLabels];
            for (int s = 0; s < numLabels; s++) {
                for (int j = 0; j < N; j++) {
                    double[] sprobs = u.getColumn(prevcurr, s);
                    int maxTag = ArrayUtil.argmax(sprobs);

                    // IF NO USAGE OF MOST LIKELY TAG ARGMAX AS NORMAL VITERBI APPROACH
                    if(labelUsageCounter[maxTag] == 0)
                    {
                        tagPointers[j][s] = maxTag;

                        vit[j][s] = sprobs[tagPointers[j][s]];
                        tagVPointers[j][s] = labelUsageCounter[maxTag];
                        labelUsageCounter[maxTag]++;
                    }
                    /*else if(i==1)
                    {
                        maxTag = u.nthLargest(j+1,sprobs);
                        tagPointers[j][s] = maxTag;

                        vit[j][s] = sprobs[tagPointers[j][s]];
                        tagVPointers[j][s] = labelUsageCounter[maxTag];
                        labelUsageCounter[maxTag]++;
                    }*/
                    // IF TAG HAS BEEN USED RECALCULATE WITH 2nd MOST LIKELY (USAGE) FOR THAT TOKEN AND TAKE THE ARGMAX AS NORMAL
                    else {
                        if (labelUsageCounter[maxTag] < N) {
                            int usage = labelUsageCounter[maxTag];

                            double[] prevCurrLine = ArrayUtil.add(prevcurr[s], tokens.get(i - 1).getData()[usage][s]);

                            sprobs[s] = prevCurrLine[s];
                            tagPointers[j][s] = ArrayUtil.argmax(sprobs);

                            vit[j][s] = sprobs[tagPointers[j][s]];
                            tagVPointers[j][s] = labelUsageCounter[maxTag];
                            labelUsageCounter[maxTag]++;
                        }

                        else {
                            System.out.println("readj");
                            boolean found = false;
                            int nextHighestFreeTag = 2;
                            while (!found) {
                                maxTag = u.nthLargest(nextHighestFreeTag, sprobs);
                                if (labelUsageCounter[maxTag] < N) {
                                    int usage = labelUsageCounter[maxTag];
                                    double[] prevCurrLine = ArrayUtil.add(prevcurr[s], tokens.get(i - 1).getData()[usage][s]);

                                    u.p(s);
                                    u.p(sprobs);

                                    sprobs[s] = prevCurrLine[s];
                                    tagPointers[j][s] = ArrayUtil.argmax(sprobs);


                                    u.p(sprobs);
                                    u.p(ArrayUtil.argmax(sprobs));

                                    vit[j][s] = sprobs[tagPointers[j][s]];
                                    tagVPointers[j][s] = labelUsageCounter[maxTag];
                                    labelUsageCounter[maxTag]++;

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

            }
            //u.p(labelUsageCounter);
            u.p(tagPointers);
            tokens.get(i).setData(vit);
            tokens.get(i).setTagPointer(tagPointers);
            tokens.get(i).setTagVersionPointer(tagVPointers);
        }

        int k = 0;

        sentence.labels[T - 1] = u.nthLargest(2,(tokens.get(T-1).getData()[k]));
        System.out.println("***" + m.labelVocab.name(sentence.labels[T - 1]));
        int backtrace = tokens.get(T-1).getTagPointer() [tokens.get(T-1).getTagVersionPointer()[k][sentence.labels[T - 1]]] [sentence.labels[T - 1]];

        for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
            sentence.labels[i] = backtrace;
            System.out.println("***" + m.labelVocab.name(backtrace));
            backtrace = tokens.get(i).getTagPointer() [tokens.get(i).getTagVersionPointer()[k][backtrace]] [backtrace];
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

        double[][] data = new double[N][numLabels];
        int[][] tagPointer = new int[N][numLabels];
        int[][] tagVersionPointer = new int[N][numLabels];

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