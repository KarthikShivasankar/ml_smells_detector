"""Applies a trained Spacy3 entity/span detection and categorization system on new data. 
It also produces and saves color-highlighted visualizations for the detected entities.
"""

import os
import spacy
import argparse

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-model_dir", type=str, help="Path to a trained Spacy 3 model", required=True)
    argparser.add_argument("-test_data", type=str, help="Path to a text file (or directly some text) to test detection.", required=True)
    argparser.add_argument("-from_file", help="Add arg if a text file path is passed in 'test_data', omit it if the text passed directly in 'test_data'", action='store_true')
    argparser.add_argument("-out_dir", type=str, help="Path to dir where to save visualizations.", default="")
    args = argparser.parse_args()

    trained_model = spacy.load(args.model_dir)
    
    # Load example text(s)
    if args.from_file:
        with open(args.test_data) as f:
            example_text = f.read()
    else:
        example_text = args.test_data
        
    # Apply trained model on data to detect entities automatically 
    doc = trained_model(example_text)
    #print("Detected entities:", doc.ents) # TO DO: print with better formatting? 
    colors = {"event_AttackDatabreach": "#5499C7",
            "event_AttackPhishing": "#5499C7",
            "event_AttackRansom": "#5499C7", 
            "event_DiscoverVulnerability": "#5499C7", 
            "event_PatchVulnerability": "#5499C7"}
    
    # Produce visualizations with color-highlighting for detected entities
    print("Generating color-highlighted visualization and saving it to disk ...")
    out = spacy.displacy.render(doc, style="ent", page="true", options = {"colors" : colors})
    with open(os.path.join(args.out_dir, "detection_result.html"), "w") as f:
        f.write(out)
    print("Done!")


    