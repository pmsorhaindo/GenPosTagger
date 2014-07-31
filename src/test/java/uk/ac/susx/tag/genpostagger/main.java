package uk.ac.susx.tag.genpostagger;

import uk.ac.susx.tag.dependencyparser.Parser;
import uk.ac.susx.tag.dependencyparser.datastructures.Sentence;
import uk.ac.susx.tag.dependencyparser.datastructures.Token;
import uk.ac.susx.tag.dependencyparser.textmanipulation.CoNLLReader;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by ps324 on 29/07/2014.
 */
public class Main {

    public static void main(String[] args) throws IOException {

        Parser p = new Parser();
        p.printHelpfileAndOptions();

        Sentence s = new Sentence();
        File f = new File("src/Main/resources/daily547.conll");
        CoNLLReader c = new CoNLLReader(f, "form, pos");

        //p.parseSentence(s);



        List<Token> result = p.parseSentence(c.next());
        for(Token x : result)
        {
            System.out.println(x.toString());
        }
    }
}
