package util;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.RandomAccessFile;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import org.apache.commons.io.FileUtils;

public class FileSplitter {

	public FileSplitter() {

	}

	public void readInAllTweetsSplit(String fileName, int numSplits, int numReps) {

		int lineCounter = 0;
		int tweetNum = 0;
		ArrayList<ArrayList<String>> tweets = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {

			String sCurrentLine;
			ArrayList<String> tweet = new ArrayList<>();
			while ((sCurrentLine = br.readLine()) != null) {

				if (sCurrentLine.length() < 2) {
					tweets.add(tweet);
					tweetNum++;
					tweet = new ArrayList<>();
				}
				tweet.add(sCurrentLine);

			}
			System.out.println("tweetNum: " + tweetNum);
		} catch (IOException e) {
			e.printStackTrace();
		}

		ArrayList<ArrayList<Integer>> indexes = new ArrayList<>();
		ArrayList<Integer> firstIndexes = new ArrayList<>();

		for (int i = 0; i < tweetNum; i++) {
			firstIndexes.add(i);
		}

		for (int i = 0; i < numReps; i++) {
			ArrayList<Integer> nextIndexes = new ArrayList<>();
			nextIndexes.addAll(firstIndexes);
			Collections.shuffle(nextIndexes, new Random(i));
			indexes.add(nextIndexes);
		}
		
		// write model files
		// for each randomization list
		for (int i = 0; i < indexes.size(); i++) {
			//System.out.println(indexes.get(i).toString());
			int splitSize = tweetNum/numSplits;
			int tweetCount = 0;
			int splitCount =0;
			for (int j= 0; j<indexes.get(i).size(); j++)
			{
				if(tweetCount>splitSize && splitCount+1 < numSplits)
				{
					splitCount ++;
					tweetCount = 0;
				}
				String sFileName = "/Volumes/LocalDataHD/ps324/data/CV2/data/Run_"+(i+1)+"/gate_split_"
						+ (splitCount+1) + ".conll";
				//System.out.println(indexes.get(i).get(j));
				ArrayList<String> lines = tweets.get(indexes.get(i).get(j));
				try {
					FileUtils.writeLines(new File(sFileName), lines, true);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				tweetCount++;
			}
		}
        System.out.println("");

	}

	public static void split(String fileName, int numSplits) {

		LineNumberReader lnr;
		int numOfLines = 0;
		try {
			lnr = new LineNumberReader(new FileReader(new File(fileName)));

			lnr.skip(Long.MAX_VALUE);
			System.out.println(lnr.getLineNumber());
			numOfLines = lnr.getLineNumber();
			// Finally, the LineNumberReader object should be closed to prevent
			// resource leak
			lnr.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		assert numOfLines > 0; // TODO make sure there are no empty files being
								// split.

		int minLinesforSplit = numOfLines / numSplits;

		FileReader fr = null;
		try {
			fr = new FileReader(fileName);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		int lineCounter = 0;
		try (BufferedReader br = new BufferedReader(fr)) {

			String sCurrentLine;
			int fileNum = 1;
			while ((sCurrentLine = br.readLine()) != null) {
				lineCounter++;
				String sFileName = "/Volumes/LocalDataHD/ps324/Documents/workspace/TwitterNLP/bin/data/daily547_"
						+ fileNum + ".conll";

				FileUtils.write(new File(sFileName), sCurrentLine + "\n",
						Charset.defaultCharset(), true);

				System.out.println(sCurrentLine);

				if (lineCounter > minLinesforSplit && sCurrentLine.length() < 1) {
					lineCounter = 0;
					fileNum++;
					System.out.println("split!");
				}

			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public void byteSplit(String filename) throws Exception {
		RandomAccessFile raf = new RandomAccessFile(filename, "r");
		long numSplits = 10; // from user input, extract it from args
		long sourceSize = raf.length();
		long bytesPerSplit = sourceSize / numSplits;
		long remainingBytes = sourceSize % numSplits;

		int maxReadBufferSize = 8 * 1024; // 8KB
		for (int destIx = 1; destIx <= numSplits; destIx++) {
			BufferedOutputStream bw = new BufferedOutputStream(
					new FileOutputStream("split_" + destIx));
			if (bytesPerSplit > maxReadBufferSize) {
				long numReads = bytesPerSplit / maxReadBufferSize;
				long numRemainingRead = bytesPerSplit % maxReadBufferSize;
				for (int i = 0; i < numReads; i++) {
					readWrite(raf, bw, maxReadBufferSize);
				}
				if (numRemainingRead > 0) {
					readWrite(raf, bw, numRemainingRead);
				}
			} else {
				readWrite(raf, bw, bytesPerSplit);
			}
			bw.close();
		}
		if (remainingBytes > 0) {
			BufferedOutputStream bw = new BufferedOutputStream(
					new FileOutputStream("split_" + numSplits + 1));
			readWrite(raf, bw, remainingBytes);
			bw.close();
		}
		raf.close();
	}

	static void readWrite(RandomAccessFile raf, BufferedOutputStream bw,
			long numBytes) throws IOException {
		byte[] buf = new byte[(int) numBytes];
		int val = raf.read(buf);
		if (val != -1) {
			bw.write(buf);
		}
	}
}