package util;

import junit.framework.TestCase;

public class UtilTest extends TestCase {

    public double[] arr = {0.0,0.1,0.2,0.3,0.4};

    public void setUp() throws Exception {
        super.setUp();

    }


    public void testNthLargestOutOfBounds() {

        Util tester = new Util();

        // Tests
        assertEquals("The 0th largest in [0.0,0.1,0.2,0.3,0.4] must be -1", -1, tester.nthLargest(0, arr));
        assertEquals("The -1th largest in [0.0,0.1,0.2,0.3,0.4] must be -1", -1, tester.nthLargest(-1, arr));
        assertEquals("The 6th largest in [0.0,0.1,0.2,0.3,0.4] must be -1", -1, tester.nthLargest(6, arr));
    }
}