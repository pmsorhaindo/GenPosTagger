package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;
import impl.decoders.IDecoder;

import java.util.ArrayList;
import java.util.Arrays;

import util.Tree;
import util.Util;
import edu.berkeley.nlp.util.ArrayUtil;

public class ViterbiArray implements IDecoder {
	
	private Model m;
	private int numLabels;
	private int N;
	private Util u;
	
	public ViterbiArray(Model m){
		
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
        Tree.Node<double[]> rootNode = new Tree.Node(vit);
        rootNode.setPointers(origPointer);
        Tree t = new Tree(rootNode);
        double[] x = new double[numLabels];
        x = (double[]) t.getRoot().getData();
        //u.p(x);
        ArrayList<Tree.Node> originalLeaf = new ArrayList<>();
        originalLeaf.add(rootNode);
        leaves.add(originalLeaf);
        //oldLeaves.add(rootNode);


        for(int token = 1; token<T; token++) {

            ArrayList<Tree.Node> newLeaves = new ArrayList<>();
            for (Tree.Node<double[]> prevPathStub: leaves.get(token-1))
            {
                double[][] prevcurr = new double[numLabels][numLabels];
                for (int s = 0; s < numLabels; s++) {
                    //System.out.println("labelScores[" + s + "]" + labelScores[0][s]);
                    computeVitLabelScores(token, s, sentence, prevcurr[s]);
                    //System.out.println("prevcurr[" + s + "] " + ArrayUtil.toString(prevcurr[s]));
                    ArrayUtil.logNormalize(prevcurr[s]);
                    prevcurr[s] = ArrayUtil.add(prevcurr[s], prevPathStub.getData());
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
                    System.out.println("*** " + n.toString());
                }
            }
            leaves.add(newLeaves);
            System.out.println("size:"+leaves.size());
            //oldLeaves.clear();
            //oldLeaves.addAll(leaves.get(token));
            //if(token != T-1) leaves.clear();

        }
        for (int i = 0; i <leaves.get(leaves.size()-1).size(); i++){ //Tree.Node<double[]> i : leaves) {
            double[] y = new double[numLabels];
            y = (double[]) leaves.get(leaves.size()-1).get(i).getData();
            u.p(y);
        }

        double[] arr = (double[]) leaves.get(T-1).get(0).getData();
        int resOne = ArrayUtil.argmax(arr);

        Tree.Node one = leaves.get(T-1).get(0);
        Tree.Node two = one.getParent();
        Tree.Node three = two.getParent();
        Tree.Node four = three.getParent();

        System.out.println("§§§§§§");
        u.p((double[]) one.getData());
        System.out.println("§§§§§§");


        u.p((int[]) one.getPointers());
        u.p((int[]) two.getPointers());
        u.p((int[]) three.getPointers());
        u.p((int[]) four.getPointers());

        int resTwo = two.getPointers()[resOne];
        int resThree = three.getPointers()[resTwo];
        int resFour = four.getPointers()[resThree];

        System.out.println("best back pointer:"+resOne+","+resTwo+","+resThree+","+resFour);


        u.p(leaves.size());
        for(int i=0; i<leaves.size();i++) {
            u.p(leaves.get(i).size());
        }
        //[14, 6, 2, 4, 1, 7]

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
