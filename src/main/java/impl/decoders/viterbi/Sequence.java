package impl.decoders.viterbi;

import impl.Model;
import impl.ModelSentence;

import java.util.ArrayList;
import java.util.Collections;

import util.Util;
import edu.berkeley.nlp.util.ArrayUtil;
import edu.stanford.nlp.util.ArrayUtils;

public class Sequence {
	
	private ArrayList<Integer> listOfTags = new ArrayList<Integer>();
	private double probabilityOfSequence;
	private ArrayList<Sequence> pathSegments = new ArrayList<Sequence>();
	
	public ArrayList<Sequence> getPathSegments() {
		return pathSegments;
	}

	public void setPathSegments(ArrayList<Sequence> pathSegments) {
		this.pathSegments = pathSegments;
	}

	public Sequence() {
		
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Sequence other = (Sequence) obj;
		if (listOfTags == null) {
			if (other.listOfTags != null)
				return false;
		} else if (!listOfTags.equals(other.listOfTags))
			return false;
		return true;
	}

	public Sequence(int startMarker, ArrayList<Double> sequenceProbs, int ... tags){
		this.listOfTags.add(startMarker);
		for(int i=0; i<tags.length; i++)
		{
			this.listOfTags.add(tags[i]);
		}
		
		ArrayList<Integer> tempTags = new ArrayList<>();
		tempTags.addAll(this.listOfTags);
		int count = sequenceProbs.size();

		this.setProbabilityOfSequence(sequenceProbs.get(sequenceProbs.size()-1));
		while(tempTags.size()>1)
		{	
			pathSegments.add(new Sequence(tempTags));
			tempTags.remove((tempTags.size()-1));
			double prob = 1;
			for (int i = 0; i<count; i++)
			{
				prob = prob + sequenceProbs.get(i);
			}
			count--;
			pathSegments.get(pathSegments.size()-1).setProbabilityOfSequence(prob);
			
			//System.out.println(pathSegments.get((pathSegments.size()-1)).listOfTags.toString() + " : " + pathSegments.get((pathSegments.size()-1)).getProbabilityOfSequence());
		}
		Collections.reverse(pathSegments);
		//System.out.println("we have "+pathSegments.size()+" segments.");
	}
	
	public Sequence(double prob, int startMarker, int ... tags){
		this.listOfTags.add(startMarker);
		for(int i=0; i<tags.length; i++)
		{
			this.listOfTags.add(tags[i]);
		}
		
		this.probabilityOfSequence = prob;
	}


	public Sequence(ArrayList<Integer> tags) {
		for(int i=0; i<tags.size(); i++)
		{
			this.listOfTags.add(tags.get(i));
		}
	}

	public Sequence(double prob, ArrayList<Integer> tags) {
		for(int i=0; i<tags.size(); i++)
		{
			this.listOfTags.add(tags.get(i));
		}
		this.probabilityOfSequence = prob;
	}

	public ArrayList<Integer> getListOfNodes() {
		return listOfTags;
	}


	public void setListOfNodes(ArrayList<Integer> listOfNodes) {
		this.listOfTags = listOfNodes;
	}


	public double getProbabilityOfSequence() {
		return probabilityOfSequence;
	}


	public void setProbabilityOfSequence(double probabilityOfSequence) {
		this.probabilityOfSequence = probabilityOfSequence;
	}
	
	public int get(int i) {
		return this.listOfTags.get(i);
	}
	
	public Sequence getIthPathSegment(int index) {
		if (index<0 || index >= listOfTags.size()) return null;// TODO check this boundary!!
		
		return pathSegments.get(index);
	}

	public void generateSegments() {
		
		ArrayList<Integer> tempTags = new ArrayList<>();
		tempTags.addAll(this.listOfTags);

		while(tempTags.size()>1)
		{	
			pathSegments.add(new Sequence(tempTags));
			tempTags.remove((tempTags.size()-1));
		}
	}

	public int[] getLabelIndexes() {
		
		//Double[] sprobsPrim = sprobsObj.toArray(new Double[sprobsObj.size()]);
		//double[] sprobs = ArrayUtils.toPrimitive(sprobsPrim);
		
		this.listOfTags.remove(0);
		Integer[] toPrim = this.listOfTags.toArray(new Integer[this.listOfTags.size()]);
		int[] tags = ArrayUtils.toPrimitive(toPrim);
		Util.p(tags);
		return tags;
	}
	
	@Override
	public String toString(){

		return this.listOfTags.toString();
	}

}
