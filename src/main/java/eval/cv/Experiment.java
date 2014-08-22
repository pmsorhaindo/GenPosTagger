package eval.cv;

import main.RunTagger;

/**
 * Created by ps324 on 13/08/2014.
 */
public class Experiment {

    // Type of decoder
    // Cross validator
    // Data set
    // Splits

    public Experiment(){

        try {
            RunTagger tagger = new RunTagger("/Volumes/LocalDataHD/ps324/IdeaProjects/GenPosTagger/src/main/resources/CV1/split_1.model",
                                             "/Volumes/LocalDataHD/ps324/IdeaProjects/GenPosTagger/src/main/resources/CV1/daily547_split_1.conll",
                                             RunTagger.Decoder.VITERBIDIV);
            tagger.finalizeOptions();
            tagger.runTagger();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

    }


    public static void main(String[] args){

        Experiment e = new Experiment();

    }


}
