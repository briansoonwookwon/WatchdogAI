import argparse
from src.detectors import PosterDetector, AI_Poster_Detector, AI_Non_Poster_Detector, ArtifactDetector

def main():
    parser = argparse.ArgumentParser(description='AI Content Detection Pipeline')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--poster_threshold', type=float, default=0.5,
                        help='Poster detection threshold (default: 0.5)')
    parser.add_argument('--ai_threshold', type=float, default=0.75,
                        help='AI detection threshold (default: 0.75)')
    args = parser.parse_args()
    
    # First stage: Poster detection
    poster_detector = PosterDetector()
    poster_pred, poster_conf = poster_detector.predict(args.image_path, args.poster_threshold)
    
    if poster_pred:
        # Second stage: AI detection for posters
        ai_poster_detector = AI_Poster_Detector()
        ai_pred, ai_conf = ai_poster_detector.predict(args.image_path, args.ai_threshold)
        
        print(f"Poster detected (confidence: {poster_conf:.3f})")
        print(f"AI generated: {'Yes' if ai_pred else 'No'} (confidence: {ai_conf:.3f})")
        if ai_pred:
            print(f"\nFinal Result:\n=== Flagged ===")
        else:
            print(f"\nFinal Result:\n=== Not Flagged ===")
    else:
        # Not a poster, check if it's AI-generated non-poster content
        print(f"No poster detected (confidence: {1 - poster_conf:.3f}). Checking if AI-generated...")
        
        ai_non_poster_detector = AI_Non_Poster_Detector()
        ai_non_poster_pred, ai_non_poster_conf = ai_non_poster_detector.predict(args.image_path, args.ai_threshold)
        
        if ai_non_poster_pred == 1:
            # Classified as real (not AI), check for artifacts
            print(f"Not detected as AI (confidence: {ai_non_poster_conf:.3f}), checking for artifacts...")
            
            artifact_detector = ArtifactDetector()
            artifact_pred = artifact_detector.predict(args.image_path)

            if artifact_pred:
                print(f"Artifacts detected")
                print(f"\nFinal Result:\n=== Flagged ===")
            else:
                print(f"No artifacts detected")
                print(f"\nFinal Result:\n=== Not Flagged ===")

        elif ai_non_poster_pred == 0:
            # Classified as AI-generated
            print(f"Detected as AI-generated (confidence: {ai_non_poster_conf:.3f})")
            print(f"\nFinal Result:\n=== Flagged ===")
        
if __name__ == "__main__":
    main()