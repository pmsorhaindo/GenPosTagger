package eval.cv;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import main.Train;

import org.apache.commons.lang3.ArrayUtils;

import util.FileSplitter;

/**
 * @author ps324
 *
 * This class produces the cross validation folds, training the models used by the Experiment class.
 * It also makes critical use of the FileSplitter class.
 *
 */

public class CrossValidator {

	public CrossValidator() {
		// produce folds
		generate();

	}
	
	private void generate(){
		File path = new File("/Volumes/LocalDataHD/ps324/data/CV4/data");
		FileSplitter splits = new FileSplitter();
		splits.readInAllTweetsSplit(
				"/Volumes/LocalDataHD/ps324/data/combCMU.conll",
				10, 2);
		trainModels(path, "comb_split_");
		kFold(path);
	}

	private void trainModels(File path, String fileName) {

		String[] directories = path.list(new FilenameFilter() {
			@Override
			public boolean accept(File current, String name) {
				return true;//new File(current, name).isDirectory();
			}
		});
        for(String s : directories)
        {
            System.out.println(s);
        }

		for (int i = 1; i < directories.length; i++) {
			File f = new File(path, directories[i]);
			// System.out.println("file " + i + ": " + f.toString());
			File files[] = f.listFiles();
			for (int j = 0; j < files.length; j++) {
				File[] filesToMerge = new File[files.length-1];
				filesToMerge = ArrayUtils.clone(files);
				System.out.println(Arrays.toString(filesToMerge));
				filesToMerge =ArrayUtils.remove(filesToMerge, j);
				System.out.println(Arrays.toString(filesToMerge));

				String[] targetParts = files[j].toString().split("Run_");
				String mergeName = targetParts[0] + "Run_" + (i + 1)
						+ "/split_" + (j + 1);
				String target = targetParts[0] + "Run_" + (i + 1) + "/split_"
						+ (j + 1) + ".model";

				mergeFiles(filesToMerge, new File(mergeName));
                String[] args = { new File(mergeName).toString(), target };

				try {
					Train.main(args);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

		}

	}

	public void kFold(File path) {
		String[] directories = path.list(new FilenameFilter() {
			@Override
			public boolean accept(File current, String name) {
				return new File(current, name).isDirectory();
			}
		});

		for (int i = 0; i < directories.length; i++) {
			File f = new File(path, directories[i]);

			File files[] = f.listFiles();
			for (int j = 0; j < files.length; j++) {

			}
		}

	}

	public static void main(String[] args) throws IOException {
		CrossValidator cv = new CrossValidator();
	}

	public void mergeFiles(File[] files, File mergedFile) {

        FileWriter fstream = null;
		BufferedWriter out = null;
		try {
			fstream = new FileWriter(mergedFile, true);
			out = new BufferedWriter(fstream);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		for (File f : files) {
			System.out.println("merging: " + f.getName());
			FileInputStream fis;
			try {
				fis = new FileInputStream(f);
				BufferedReader in = new BufferedReader(new InputStreamReader(
						fis));

				String aLine;
				while ((aLine = in.readLine()) != null) {
					out.write(aLine);
					out.newLine();
                }

				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		try {
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
