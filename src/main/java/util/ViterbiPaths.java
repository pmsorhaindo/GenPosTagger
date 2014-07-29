package util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

public class ViterbiPaths {
	
	private ArrayList<ArrayList<Integer>> paths;
	private ArrayList<Double> probs;
	private Util u = new Util();	
	
	public ViterbiPaths() {
		paths = new ArrayList<ArrayList<Integer>>();
		probs = new ArrayList<Double>();
	}

	public ArrayList<ArrayList<Integer>> getPaths() {
		return paths;
	}

	public void setPaths(ArrayList<ArrayList<Integer>> paths) {
		this.paths = paths;
	}
	
	public void addPaths(int[][] viterbiPaths)
	{
		for(int i = 0; i<viterbiPaths.length; i++)
		{
			ArrayList<Integer> arr = new ArrayList<Integer>();
			paths.add(arr);
			for(int j =0; j<viterbiPaths[i].length; j++)
			{
				
				//System.out.println("viterbi path length: "+viterbiPaths[i].length);
				paths.get(paths.size()-1).add(viterbiPaths[i][j]);
			}
		}
	}

	public ArrayList<Double> getProbs() {
		return probs;
	}

	public void setProbs(ArrayList<Double> probs) {
		this.probs = probs;
	}

	public void addProbs(double[] probs) {
		Double[] doubleArray = ArrayUtils.toObject(probs);
		List<Double> list = Arrays.asList(doubleArray);
		this.probs.addAll(list);
		
	}
	
// Attempt to do this nicely avoiding type erasure with Generics
//
//	public<T, P> P ArrayListToPrimitive(ArrayList<T> arrObj, T[] objType, P primType) {
//		T[] arrOfObjs = (T[]) arrObj.toArray(objType);
//		P arr = ArrayUtils.toPrimitive(arrOfObjs);
//		
//		return arr;
//	}
	
	public double[] topNHighestProbabilities(int n) {
		assert(n < this.probs.size());
		double[] limitedProbArr = new double[n];
		for(int i=0; i<n; i++)
		{
			double[] probArr = arrayListToPrimitiveDouble(this.probs);
			int index = u.nthLargest((i+1), probArr);
			limitedProbArr[i] = this.probs.get(index);
		}
		
		return limitedProbArr;
	}
	
	public int[][] topNHighestPaths(int n, int pathSize){
		assert(n < this.probs.size());
		int[][] limitedPathArr = new int[n][pathSize];
		for(int i=0; i<n; i++)
		{
			double[] probArr = arrayListToPrimitiveDouble(this.probs);
			int index = u.nthLargest((i+1), probArr);
			//System.out.println("path "+i+": "+this.paths.get(index).toString());
			limitedPathArr[i] = arrayListToPrimitiveInt(this.paths.get(index));
		}
		//System.out.println("size of path: "+ pathSize);
		return limitedPathArr;
	}
	
	
	public int[] arrayListToPrimitiveInt(ArrayList<Integer> arrObj)
	{
		Integer[] arrOfObjs = (Integer[]) arrObj.toArray(new Integer[0]);
		int[] arr = ArrayUtils.toPrimitive(arrOfObjs);
		
		return arr;
	}

	public double[] arrayListToPrimitiveDouble(ArrayList<Double> arrObj)
	{
		Double[] arrOfObjs = (Double[]) arrObj.toArray(new Double[0]);
		double[] arr = ArrayUtils.toPrimitive(arrOfObjs);
		
		return arr;
	}

	

}
