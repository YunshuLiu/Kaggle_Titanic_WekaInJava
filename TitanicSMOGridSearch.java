import java.io.File;
import java.util.Random;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

//import weka.classifiers.Classifier;  // uncomment this if want to define classifier using <xxx.newInstance();>
import weka.classifiers.functions.SMO; //need this if want to define classifier using <SMO cfs = new SMO();>
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.meta.GridSearch;
//import weka.filters.AllFilter;
//import weka.core.SelectedTag;

/**
 *
 * @author SkyLibrary
 */
public class TitanicSMOGridSearch {
	
    /**
     * @param args
     */
    public static void main(String[] args) {
		// TODO Auto-generated method stub
		Instances trainIns = null;
		Instances testIns = null;
		
		try{
			
			/*
			 * 1.read in training data and testing data
			 in windows: File file= new File("C://Program Files//Weka-3-6//data//contact-lenses.arff");
			 */
			
			File file= new File("trainV4_rmParch.arff");
			ArffLoader loader = new ArffLoader();
			loader.setFile(file);
			trainIns = loader.getDataSet();
                        
                        file = new File("testV4_rmParch.arff");
                        loader.setFile(file);
                        testIns = loader.getDataSet();                   
			
			/*
                         * You must set the classIndex(Labels) first, 
                         * 0 means the lable is in column 1, trainIns.numAttributes()-1 means last column
                         */
			trainIns.setClassIndex(0);
                        testIns.setClassIndex(0);
                        //trainIns.setClassIndex(trainIns.numAttributes()-1); set to this if label is the last one
                        //testIns.setClassIndex(testIns.numAttributes()-1);
                        
                        
			/*
			 * 2.initialize the classifier, and set options
			 */                        
                        
                        SMO cfs = new SMO();	
                        //cfs.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P  1.0E-12 -N 0 -V -1 -W 1 -K \"  weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E  1.0\""));
                        //cfs.setOptions(weka.core.Utils.splitOptions("-K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\" "));
                        
                        GridSearch cvgridsearch = new GridSearch();
                        
                        cvgridsearch.setOptions(weka.core.Utils.splitOptions("-E ACC -filter weka.filters.AllFilter -D " //add -D if want to print debug output
                                + "-y-property classifier.kernel.gamma -y-min -4 -y-max 0.5 -y-step 0.25 -y-base 10.0 -y-expression pow(BASE,I) "
                                + "-x-property clasifier.c -x-min 1.0 -x-max 201.0 -x-step 20 -x-base 10.0 -x-expression I "
                                + "-sample-size 100.0 -traversal COLUMN-WISE -log-file /Users/SkyLibrary/123 -num-slots 1 -S 1 "
                                + "-W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 "
                                + "-K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
                        
                        //*turn off the debuging by removing -D; pow(BASE,I) means stepSize(I) = BASE^I
                        /*
                        cvgridsearch.setOptions(weka.core.Utils.splitOptions("-E ACC -filter weka.filters.AllFilter " //add -D if want to print debug output
                                + "-y-property classifier.kernel.gamma -y-min -0.8 -y-max -0.2 -y-step 0.1 -y-base 10.0 -y-expression pow(BASE,I) "
                                + "-x-property clasifier.c -x-min 10.0 -x-max 100.0 -x-step 10 -x-base 1.0 -x-expression I "
                                + "-sample-size 100.0 -traversal COLUMN-WISE -log-file /Users/SkyLibrary/123 -num-slots 1 -S 1 "
                                + "-W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 "
                                + "-K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
                         */
                        
                        
                        /* the following options can compile but not run, probabily because some of the 
                         * valuables must be set simultaneously
                        AllFilter nofilter = new AllFilter();
                        cvgridsearch.setFilter(nofilter);
                        SelectedTag tag1 = new SelectedTag(GridSearch.EVALUATION_ACC,GridSearch.TAGS_EVALUATION);
                        cvgridsearch.setEvaluation(tag1);
                        //* different evaluation rules
                        //*Correlation coefficient (= CC)         EVALUATION_CC = 0
                        //*Root mean squared error (= RMSE)       EVALUATION_RMSE = 1
                        //*Root relative squared error (= RRSE)   EVALUATION_RRSE = 2
                        //*Mean absolute error (= MAE)            EVALUATION_MAE = 3
                        //*Root absolute error (= RAE)            EVALUATION_RAE = 4
                        //*Combined: (1-abs(CC)) + RRSE + RAE     EVALUATION_COMBINED = 5
                        //*Accuracy (= ACC)                       EVALUATION_ACC = 6
                         
                        cvgridsearch.setXProperty("classifier.c");
                        cvgridsearch.setXMin(1.0);
                        cvgridsearch.setXMax(10.0);
                        cvgridsearch.setXStep(1);
                        cvgridsearch.setXBase(10.0);
                        cvgridsearch.setXExpression("I");
                        
                        cvgridsearch.setYProperty("classifier.kernel.gamma");
                        cvgridsearch.setYMin(-2.0);
                        cvgridsearch.setYMax(1.0);
                        cvgridsearch.setYStep(0.5);
                        cvgridsearch.setYBase(10.0);
                        cvgridsearch.setYExpression("pow(BASE,I)");
                     
                        cvgridsearch.setClassifier(cfs); //* setting classifier must be after setup of gridsearch
                        */
                        
                        
			/*
			 * 3.Training the model using training dataset
			 */
                        System.out.println("\nTraining...");
			
			//*set random seed
			//gridSearch.setSeed(2);
                        
			//*train cvgridsearch classifier
			cvgridsearch.buildClassifier(trainIns);
			System.out.println("Training finished!");
                        
                        //*print out the best value find by grid search
                        System.out.println("Print out getValues");
                        System.out.println(cvgridsearch.getValues());
                        
                        //*print out best classifer, nothing is print out because this is for saving the model
                        //System.out.println("Print out getBestClassifier");
                        //System.out.println(cvgridsearch.getBestClassifier());
                        
                        //*print out the String about detailed gridsearch result, also includes support vector and number of kernels evaluation
                        System.out.println("Print out toString");
                        System.out.println(cvgridsearch.toString());
                        
                        //* pring out the SummaryString, best filter and best classifier 
                        //System.out.println("Print out toSummaryString");
                        //System.out.println(cvgridsearch.toSummaryString());
                        
                        //* If you want to read in a model, using the following code to read in the deserialize model
                        //Classifier cfs = (Classifier) weka.core.SerializationHelper.read("TitanicSMOVoting.model");
			
			/*
			 * 4. Evaluation: Class for evaluating machine learning models
			 */	
                        //*print classifier detail
                        //System.out.println(cvgridsearch); 
                        
                        //* 10 fold cross validation on trainset
                        //* use Random(1) as random seed to checkout the Evaluation
                        System.out.println("Running cross validation on training set...");
                        Evaluation eval_crossValid = new Evaluation(trainIns);
                        eval_crossValid.crossValidateModel(cvgridsearch, trainIns,10,new Random(1));
                        System.out.println(eval_crossValid.toSummaryString("\nResults\n======\n", false));                       
                        
                        //* the following setting automaticly run 10 fold crossValidation and also will give detailed statistics.
			//System.out.println("Running cross validation and get detailed statistics...");
                        //String[] options = new String[2]; 
                        //options[0] = "-t";  
                        //options[1] = "trainV4_rmParch.arff";                        
                        //System.out.println(Evaluation.evaluateModel(cvgridsearch, options));                        
                        
                        //* use test set to verify the result.
                        //System.out.println("Verify model with test set...");
                        //Evaluation eval = new Evaluation(trainIns);
                        //eval.evaluateModel(cvgridsearch, testIns);
                        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                        

 
                        
			/*
			 * 5. output generated model
			 */  
                        weka.core.SerializationHelper.write("TitanicSMOGridSearch.model", cvgridsearch);
                        
                        
                        
                        
			/*
			 * 6. predict on test set and output result in csv
                         * If you're interested in the distribution over all the classes, use the method distributionForInstance(Instance). 
                         * This method returns a double array with the probability for each class.
			 */                       
                        System.out.println("Make prediction on unlabled data and write to file...");
                         //* load unlabeled data
                         Instances unlabeled = new Instances(new BufferedReader(new FileReader("testV4_rmParch.arff")));
 
                         //* set class attribute
                         unlabeled.setClassIndex(0);
 
                         //* create copy
                         Instances labeled = new Instances(unlabeled);
 
                         //* label instances
                         for (int i = 0; i < unlabeled.numInstances(); i++) {
                            double clsLabel = cvgridsearch.classifyInstance(unlabeled.instance(i));
                            labeled.instance(i).setClassValue(clsLabel);
                          }
                          //* save labeled data
                         BufferedWriter writer = new BufferedWriter(new FileWriter("Predicted.csv"));
                         writer.write(labeled.toString());
                         writer.newLine();
                         writer.flush();
                         writer.close();
                         System.out.println("Finish writing to file!");
                         
		}catch(Exception e){
			e.printStackTrace();
		}
    }

}
