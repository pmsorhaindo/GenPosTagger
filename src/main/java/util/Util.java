package util;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.nlp.util.StringUtils;


public class Util { 
	public static void p(Object x) { System.out.println(x); }
	public static void p(String[] x) { p(Arrays.toString(x)); }
	public static void p(double[] x) { p(Arrays.toString(x)); }
	public static void p(int[] x) { p(Arrays.toString(x)); }
	public static void p(double[][] x) {

		System.out.printf("(%s x %s) [\n", x.length, x[0].length);
		for (double[] row : x) {
			System.out.printf(" ");
			p(Arrays.toString(row));
		}
		p("]");
	}
	public static void p(int[][] x) {

		System.out.printf("(%s x %s) [\n", x.length, x[0].length);
		for (int[] row : x) {
			System.out.printf(" ");
			p(Arrays.toString(row));
		}
		p("]");
	}
	public static String sp(double[] x) {
		ArrayList<String> parts = new ArrayList<String>();
		for (int i=0; i < x.length; i++)
			parts.add(String.format("%.2g", x[i]));
		return "[" + StringUtils.join(parts) + "]";
	}
	//	public static void p(int[][] x) { p(Arrays.toString(x)); }
	public static void p(String x) { System.out.println(x); }

	public void squeeze(double val,int position,double[] squeezedArr)
	{
		boolean endReached = false;
		double hold = 0;
		double nextVal = val; 
		if(position>squeezedArr.length) endReached = true;
		
		while(!endReached)
		{
			hold = squeezedArr[position];
			squeezedArr[position] = nextVal;
			nextVal = hold;
			position ++;
			if (position>=squeezedArr.length) endReached = true;
		}
	}
	
	
	public int nthLargest(int n, double[] inputArr){

        if(n<=0 || n>inputArr.length) return -1;

		double[] record = new double[n];
		Arrays.fill(record, Double.NEGATIVE_INFINITY);
		
		for(int i = 0; i<inputArr.length; i++)
		{
			for (int j = 0; j<record.length; j++)
			{
				if(inputArr[i]>record[j])
				{
					//System.out.println("Squeeze record " + inputArr[i]);
					squeeze(inputArr[i],j,record);
					//p(record);
					break;
				}
			}
		}
		
		for (int i =0;i<inputArr.length; i++)
		{
			if(record[n-1]==inputArr[i])
			{
				return i;
			}
		}
		return 0;
	}
	
	public double[] getColumn(double[][] matrix, int col) {
		double[] column = new double[matrix.length];
		for (int i = 0; i < matrix[0].length; i++) {
			column[i] = matrix[i][col];
		}
		return column;
	}
}