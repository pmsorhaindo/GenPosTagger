package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;

import util.Tree;
import util.Util;
import edu.berkeley.nlp.util.ArrayUtil;

public class ViterbiArrayTree implements IDecoder {
	
	private Model m;
	private int numLabels;
	private int N;
	private Util u;
	
	public ViterbiArrayTree(Model m){
		
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
		int branchLimit = 4;
		sentence.labels = new int[T]; // final labeled sequence

        double[] vit = new double[numLabels]; // viterbi Matrix

        double[][] labelScores = new double[((int) Math.pow(N,T))][numLabels];
        Tree.Node[] frontier =  new Tree.Node[branchLimit]; //TODO store edge nodes in an fixed length array and only progress these nodes,
		ArrayList<ArrayList<Tree.Node>> leaves = new ArrayList<>();
        //ArrayList<Tree.Node> oldLeaves = new ArrayList<>();

        int[] origPointer = new int[numLabels];
        for (int k = 0; k < numLabels; k++) {
            origPointer[k] = m.startMarker();
        }

		computeVitLabelScores(0, m.startMarker(), sentence, labelScores[0]);
		ArrayUtil.logNormalize(labelScores[0]);

        // Assigning first label scores to the first column in the viterbi matrix
		vit = labelScores[0];
        Tree.Node<double[]> rootNode = new Tree.Node();
        rootNode.setData(vit,origPointer);
        Tree t = new Tree(rootNode);

        ArrayList<Tree.Node> originalLeaf = new ArrayList<>();
        originalLeaf.add(rootNode);
        leaves.add(originalLeaf);


        for(int token = 1; token<T; token++) {

            ArrayList<Tree.Node> newLeaves = new ArrayList<>();
            for (Tree.Node<double[]> prevPathStub : leaves.get(token-1))
            {
                double[][] prevcurr = new double[numLabels][numLabels];
                for (int s = 0; s < numLabels; s++) {
                    //System.out.println("labelScores[" + s + "]" + labelScores[0][s]);
                    computeVitLabelScores(token, s, sentence, prevcurr[s]);
                    //System.out.println("prevcurr[" + s + "] " + ArrayUtil.toString(prevcurr[s]));
                    ArrayUtil.logNormalize(prevcurr[s]);
                    prevcurr[s] = ArrayUtil.add(prevcurr[s], prevPathStub.getData()[s]);
                }

                for (int i = 0; i < this.N; i++) {
                    double[] newVit = new double[numLabels];
                    int[] newBptr = new int[numLabels];
                    for (int s = 0; s < numLabels; s++) {
                        double[] sprobs = u.getColumn(prevcurr, s);
                        newBptr[s] = u.nthLargest((i + 1), sprobs);
                        newVit[s] = sprobs[newBptr[s]]; //ArrayUtil.argmax(sprobs)]);
                    }
                    //labelScores[s] = newVit;
                    Tree.Node n = new Tree.Node();
                    n.setData(newVit, newBptr);
                    prevPathStub.addChild(n);
                    newLeaves.add(n);
                    System.out.println("--- " + ArrayUtil.argmax((double[]) n.getData()));
                    System.out.println("*** " + Arrays.toString(n.getPointers()));
                }
            }
            leaves.add(newLeaves);
            System.out.println("size:"+newLeaves.size());

        }
//        for (int i = 0; i <leaves.get(leaves.size()-1).size(); i++){ //Tree.Node<double[]> i : leaves) {
//            double[] y = new double[numLabels];
//            y = (double[]) leaves.get(leaves.size()-1).get(i).getData();
//            u.p(y);
//        }


        Tree.Node[] nodes = new Tree.Node[T-1];
        nodes[0] = leaves.get(T-1).get(0);
        for (int i=1;i<T-1;i++)
        {
            nodes[i] = nodes[i-1].getParent();
        }
        int[] labels = new int[T-1];
        labels[0] = ArrayUtil.argmax((double[]) nodes[0].getData());
        for (int i=1;i<T-1;i++)
        {
            labels[i] = nodes[i].getPointers()[i-1];
        }

//        Tree.Node one = leaves.get(T-1).get(0);
//        Tree.Node two = one.getParent();
//        Tree.Node three = two.getParent();
//        Tree.Node four = three.getParent();
//        Tree.Node five = four.getParent();
//        Tree.Node six = five.getParent();
//        Tree.Node seven = six.getParent(); // NULL
//
//        int resOne = ArrayUtil.argmax((double[]) one.getData());
//        int resTwo = one.getPointers()[resOne];
//        int resThree = two.getPointers()[resTwo];
//        int resFour = three.getPointers()[resThree];
//        int resFive = four.getPointers()[resFour];
//        int resSix = five.getPointers()[resFive];
//        int resSeven = six.getPointers()[resSix];
//
//        System.out.println("best back pointer:"+m.labelVocab.name(resOne)+resOne+","+m.labelVocab.name(resTwo)+resTwo+","+m.labelVocab.name(resThree)+resThree+","
//                +m.labelVocab.name(resFour)+resFour+","+m.labelVocab.name(resFive)+resFive+","+m.labelVocab.name(resSix)+resSix+","+resSeven);
        System.out.println(Arrays.toString(labels));
        sentence.labels = labels;

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

}
