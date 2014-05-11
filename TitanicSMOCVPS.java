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
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Utils;
/**
 *
 * @author SkyLibrary
 */
public class TitanicSMOCVPS {

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
                        //cfs.setOptions(weka.core.Utils.splitOptions("-C 2.0"));
                        //cfs.setOptions(weka.core.Utils.splitOptions("-K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\" "));
                        cfs.setOptions(weka.core.Utils.splitOptions("-K \"  weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E  1.0\""));
                        
                        CVParameterSelection cvps = new CVParameterSelection();
                        cvps.setClassifier(cfs);
                        cvps.setNumFolds(5);  // using 5-fold CV
                        cvps.addCVParameter("C 1 100 15");
			//*set random seed
			cvps.setSeed(2);
                        
                        
			/*
			 * 3.Training the model using training dataset
			 */
                        System.out.println("\nTraining...");
			

			//*training cvps classifier
			cvps.buildClassifier(trainIns);
			System.out.println("Training finished!");
			System.out.println(Utils.joinOptions(cvps.getBestClassifierOptions()));
                        //* If you want to read in a model, using the following code
                        //* deserialize model
                        //Classifier cfs = (Classifier) weka.core.SerializationHelper.read("TitanicSMOVoting.model");
			
			/*
			 * 4. Evaluation: Class for evaluating machine learning models
			 */	
                        //*print classifier detail
                        //System.out.println(cvps); 
                        
                        //* 10 fold cross validation on trainset
                        //* use Random(1) as random seed to checkout the Evaluation
                        System.out.println("Running cross validation on training set...");
                        Evaluation eval_crossValid = new Evaluation(trainIns);
                        eval_crossValid.crossValidateModel(cvps, trainIns,10,new Random(1));
                        System.out.println(eval_crossValid.toSummaryString("\nResults\n======\n", false));                       
                        
                        //* the following setting automaticly run 10 fold crossValidation and also will give detailed statistics.
			//System.out.println("Running cross validation and get detailed statistics...");
                        //String[] options = new String[2]; 
                        //options[0] = "-t";  
                        //options[1] = "trainV4_rmParch.arff";                        
                        //System.out.println(Evaluation.evaluateModel(cvps, options));                        
                        
                        //* use test set to verify the result.
                        //System.out.println("Verify model with test set...");
                        //Evaluation eval = new Evaluation(trainIns);
                        //eval.evaluateModel(cvps, testIns);
                        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                        

 
                        
			/*
			 * 5. output generated model
			 */  
                        weka.core.SerializationHelper.write("TitanicSMOGridSearch.model", cvps);
                        
                        
                        
                        
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
                            double clsLabel = cvps.classifyInstance(unlabeled.instance(i));
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
