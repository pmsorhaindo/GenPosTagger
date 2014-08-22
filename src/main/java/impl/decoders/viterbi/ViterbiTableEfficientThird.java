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

        this.K = 1;
        this.u = new Util();
        this.m = m;
        this.numLabels = m.labelVocab.size();
        assert numLabels != 0;

    }

    @Override
    public void decode(ModelSentence sentence) {
        sentence.K = this.K;
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
            double[][] vit = new double[K][numLabels]; // should be [K][T][numLabels] no?
            int[][] tagPointers = new int[K][numLabels];
            int[][] tagVPointers = new int[K][numLabels];
            double[][] sprobs = new double[numLabels][numLabels];

            double[][] origprevcurr = new double[numLabels][numLabels];

            for (int s = 0; s < numLabels; s++) {
                computeVitLabelScores(i, s, sentence, prevcurr[s]);
                ArrayUtil.logNormalize(prevcurr[s]);
                origprevcurr[s] = prevcurr[s].clone();
                prevcurr[s] = ArrayUtil.add(prevcurr[s], tokens.get(i-1).getData()[0][s]);
            }

            for(int s = 0; s<numLabels; s++)
            {
                sprobs[s] = u.getColumn(prevcurr, s);
            }

            int[][] labelUsageCounter = new int[numLabels][numLabels];
            for (int j = 0; j < K; j++) {

                for (int s = 0; s < numLabels; s++) {

                    boolean found = false;
                    int nextHighestFreeTag = 1;
                    while (!found) {

                        // nextHighestFreeTag vs MaxTag
                        int maxTag = u.nthLargest(nextHighestFreeTag, sprobs[s]); // determine the tag with highest probability
                        if (maxTag != -1 && labelUsageCounter[nextHighestFreeTag-1][maxTag] < K) { // if no error and highest tag hasn't been overused (K), continue

                            int  usage = labelUsageCounter[s][maxTag]; // how many times the maximum tag has been used.
                            double[] prevCurrLine = ArrayUtil.add(origprevcurr[s], tokens.get(i - 1).getData()[usage][maxTag]);

                            if(usage != 0) // usage != 0 ||  if the maximum tag has been used before swap in the new confidence value in a transition to this tag.
                            {
                                sprobs[s][maxTag] = prevCurrLine[maxTag];
                            }

                            if(labelUsageCounter[s][ArrayUtil.argmax(sprobs[s])]>=K) // if the new max has hit max usage skip to next iteration.
                            {
                                nextHighestFreeTag++; // make sure to referr to the next highest max in future
                                continue;
                            }

                            tagPointers[j][s] = ArrayUtil.argmax(sprobs[s]); // back pointer to tag
                            tagVPointers[j][s] = labelUsageCounter[s][tagPointers[j][s]]; // back pointer to version of tag.

                            vit[j][s] = sprobs[s][tagPointers[j][s]]; // assigning data this is not overwriting due to the previous viterbi data being held in the previous WordData object.

                            labelUsageCounter[s][tagPointers[j][s]]++; //increment usage counter for the token whose transition we have used.
                            //u.p(labelUsageCounter);
                            found = true; // indicate to the while loop we have found our next most likely transition probability.
                        } else if (nextHighestFreeTag == numLabels) { // if we run out out tags throw an error.
                            System.err.println("Ran out of Tags");
                            break; // break from while loop in case of error.
                        } else {
                            nextHighestFreeTag++; // assuming an issue such as labelUsageCounter being over K for the current token look for the next most likely tag.
                        }
                    }
                }
            }
            //u.p(labelUsageCounter);
            u.p(tagPointers);
            u.p(vit);
            //u.p(tagVPointers);
            tokens.get(i).setData(vit);
            tokens.get(i).setTagPointer(tagPointers);
            tokens.get(i).setTagVersionPointer(tagVPointers);
        }

        sentence.nPaths = new int[K][sentence.T];
        double[][] confs = new double[K][sentence.T];
        double[] pathconfs = new double[K];

        ArrayList<ArrayList<Double>> probs = new ArrayList<>();
        for(int k = 0; k < this.K; k++) {
            probs.add(new ArrayList<Double>());
            sentence.labels[T - 1] = ArrayUtil.argmax(tokens.get(T - 1).getData()[k]);
            sentence.nPaths[k][T-1] = ArrayUtil.argmax(tokens.get(T - 1).getData()[k]);
            System.out.print("***" + m.labelVocab.name(sentence.labels[T - 1]));
            double prob = tokens.get(T - 1).getData()[k][sentence.labels[T - 1]]; //Math.exp(vit[T - 1][sentence.labels[T - 1]]);
            System.out.println(" with prob: " + prob);
            probs.get(k).add(prob);
            int vPointer = tokens.get(T - 1).getTagVersionPointer()[k][sentence.labels[T - 1]];
            int backtrace = tokens.get(T - 1).getTagPointer()[k][sentence.labels[T - 1]];

            for (int i = T - 2; (i >= 0) && (backtrace != m.startMarker()); i--) { // termination
                sentence.labels[i] = backtrace;
                sentence.nPaths[k][i] = backtrace;
                //System.out.println("***" + m.labelVocab.name(backtrace));
                double newProb = tokens.get(i).getData()[vPointer][backtrace];
                probs.get(k).add(newProb);
                System.out.println("***" + m.labelVocab.name(backtrace)
                        + " with prob: " + newProb);
                vPointer = tokens.get(i).getTagVersionPointer()[vPointer][backtrace];
                backtrace = tokens.get(i).getTagPointer()[vPointer][backtrace];
            }
            System.out.println();
            //sentence.nPaths[k] = sentence.labels;
            double totalLogProb = 0;
            for(int i = 0; i<probs.get(probs.size()-1).size(); i++) {
                double lastLogProb = probs.get(probs.size()-1).get(i);
                totalLogProb += lastLogProb;
                confs[k][i]=Math.exp(lastLogProb);
            }
            pathconfs[k] = Math.exp(totalLogProb);
        }
        sentence.confidences = confs;
        sentence.pathConfidences = pathconfs;

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