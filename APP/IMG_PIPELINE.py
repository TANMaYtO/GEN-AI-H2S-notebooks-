from TEXT_PIPELINE import run_text_pipeline

def run_img_pipeline(img_pth: str, state: dict):
    """
    Orchestrates the image analysis workflow using pre-loaded tools.
    """
    # Unpack the image-specific tools
    manipulation_analyzer = state['manipulation_analyzer']
    ocr_analyzer = state['ocr_analyzer']

    # Run the analyses
    manipulation_results = manipulation_analyzer.run_image_forensics(img_pth)
    in_image_report = ocr_analyzer.get_in_image_anal(img_pth)
    rev_img_search_res = ocr_analyzer.rev_img_search(img_pth)
    
    text_analysis_report = {}
    
    # If text is found, run the text pipeline, passing all the necessary state
    if in_image_report.get("Extracted Text", "").strip():
        text_analysis_report = run_text_pipeline(in_image_report["Extracted Text"], state)
        
    return {
        'image_manipulation_report': manipulation_results,
        'in_image_content_report': in_image_report,
        'reverse_image_search_report': rev_img_search_res,
        'extracted_text_analysis_report': text_analysis_report
    }

