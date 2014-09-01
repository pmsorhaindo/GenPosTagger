package eval.cv;

import main.RunTagger;
import org.apache.commons.io.FileUtils;

import java.io.File;

/**
 *
 * This class is is responsible for running the cross validation experiments.
 * A RunTagger constructor indicating which model file to use in order to tag which testing split.
 *
 * Created by ps324 on 13/08/2014.
 */
public class Experiment {

    // Type of decoder
    // Best K
    // Data set
    // Splits

    public Experiment(){

        for(int j =1; j<25; j++ ) {
            for (int i = 0; i < 10; i++) {
                try {
                    int splitVal = i + 1;
                    RunTagger tagger = new RunTagger("/Volumes/LocalDataHD/ps324/IdeaProjects/GenPosTagger/src/main/resources/CV1/split_" + splitVal + ".model",
                            "/Volumes/LocalDataHD/ps324/IdeaProjects/GenPosTagger/src/main/resources/CV1/daily547_split_" + splitVal + ".conll",
                            "/Volumes/LocalDataHD/ps324/exp7EffDailySent.csv",
                            RunTagger.Decoder.VITERBIEFF, j);
                    tagger.finalizeOptions();
                    tagger.runTagger();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            try {
                //FileUtils.writeStringToFile(new File("/Volumes/LocalDataHD/ps324/exp4DivComb.csv"), "\n\n\n\n", true);
            }
            catch(Exception e)
            {
                System.err.println("File access error");
            }
        }
    }


    public static void main(String[] args){

        Experiment e = new Experiment();

    }


}
