

import java.io.File;
import java.util.Random;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.Classifier;
//////import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.core.Instances;
//import weka.core.Instance;
import weka.core.converters.ArffLoader;
import weka.classifiers.meta.Vote;
import weka.core.SelectedTag;


public class TitanicSMOVoting {
	
    /**
     * @param args
     */
    public static void main(String[] args) {
		// TODO Auto-generated method stub
		Instances trainIns = null;
		Instances testIns = null;
		
                Classifier cfs1 = null;
                Classifier cfs2 = null;
                Classifier cfs3 = null;
                Classifier cfs4 = null;
                Classifier cfs5 = null;
                Classifier[] cfsArray = new Classifier[5];
		//Classifier cfs = null;
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
                        
                        //SMO cfs = new SMO();	
			cfs1 = (Classifier)Class.forName("weka.classifiers.bayes.NaiveBayes").newInstance();
                        cfs2 = (Classifier)Class.forName("weka.classifiers.functions.SMO").newInstance();
                        cfs3 = (Classifier)Class.forName("weka.classifiers.bayes.BayesNet").newInstance();
                        cfs4 = (Classifier)Class.forName("weka.classifiers.functions.SGD").newInstance();
                        cfs5 = (Classifier)Class.forName("weka.classifiers.functions.SGD").newInstance();
                        
                        //cfs.setOptions(weka.core.Utils.splitOptions("-C 3.0 -L 0.0010 -P  1.0E-12 -N 0 -V -1 -W 1 -K \"  weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E  1.0\""));
                        //String[] options = weka.core.Utils.splitOptions("-C 1");
                        //cfs.setOptions(options);
                        
                        
                        cfsArray[0] = cfs1;
			cfsArray[1] = cfs2;
			cfsArray[2] = cfs3;
                        cfsArray[3] = cfs4;
			cfsArray[4] = cfs5;
                        Vote ensemble = new Vote();
			/*
			 * Rules for ensemble classifier：
			 * AVERAGE_RULE
			 * PRODUCT_RULE
			 * MAJORITY_VOTING_RULE
			 * MIN_RULE
			 * MAX_RULE
			 * MEDIAN_RULE
			 */
			SelectedTag tag1 = new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES);
			
                     
			/*
			 * 3.Training the model using training dataset
			 */
                        System.out.println("\nTraining...");
			ensemble.setCombinationRule(tag1);
			ensemble.setClassifiers(cfsArray);
			//set random seed
			ensemble.setSeed(2);
			//training ensemble classifier
			ensemble.buildClassifier(trainIns);
			System.out.println("Training finished!");
			
                        //* If you want to read in a model, using the following code
                        //* deserialize model
                        //Classifier cfs = (Classifier) weka.core.SerializationHelper.read("TitanicSMOVoting.model");
			
			/*
			 * 4. Evaluation: Class for evaluating machine learning models
			 */	
                        //*print classifier detail
                        //System.out.println(ensemble); 
                        
                        //* 10 fold cross validation on trainset
                        //* use Random(1) as random seed to checkout the Evaluation
                        System.out.println("Running cross validation on training set...");
                        Evaluation eval_crossValid = new Evaluation(trainIns);
                        eval_crossValid.crossValidateModel(ensemble, trainIns,10,new Random(1));
                        System.out.println(eval_crossValid.toSummaryString("\nResults\n======\n", false));                       
                        
                        //* the following setting automaticly run 10 fold crossValidation and also will give detailed statistics.
			//System.out.println("Running cross validation and get detailed statistics...");
                        //String[] options = new String[2]; 
                        //options[0] = "-t";  
                        //options[1] = "trainV4_rmParch.arff";                        
                        //System.out.println(Evaluation.evaluateModel(ensemble, options));                        
                        
                        //* use test set to verify the result.
                        //System.out.println("Verify model with test set...");
                        //Evaluation eval = new Evaluation(trainIns);
                        //eval.evaluateModel(ensemble, testIns);
                        //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                        

 
                        
			/*
			 * 5. output generated model
			 */  
                        weka.core.SerializationHelper.write("TitanicSMOVoting.model", ensemble);
                
                        
                        
                        
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
                            double clsLabel = ensemble.classifyInstance(unlabeled.instance(i));
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
