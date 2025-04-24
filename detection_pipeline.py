import argparse
from src.detectors import PosterDetector, AIDetector

def main():
    parser = argparse.ArgumentParser(description='AI Content Detection Pipeline')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--poster_threshold', type=float, default=0.5,
                        help='Poster detection threshold (default: 0.5)')
    parser.add_argument('--ai_threshold', type=float, default=0.5,
                        help='AI detection threshold (default: 0.5)')
    args = parser.parse_args()
    
    # First stage: Poster detection
    poster_detector = PosterDetector()
    poster_pred, poster_conf = poster_detector.predict(args.image_path, args.poster_threshold)
    
    if poster_pred:
        # Second stage: AI detection
        ai_detector = AIDetector()
        ai_pred, ai_conf = ai_detector.predict(args.image_path, args.ai_threshold)
        
        print(f"Poster detected (confidence: {poster_conf:.3f})")
        print(f"AI generated: {'Yes' if ai_pred else 'No'} (confidence: {ai_conf:.3f})")
        if ai_pred:
            print(f"\nFinal Result:\n=== Flagged ===")
        else:
            print(f"\nFinal Result:\n=== Not Flagged ===")
    else:
        print(f"No poster detected (confidence: {1 - poster_conf:.3f}). Sending to next stage...")

if __name__ == "__main__":
    main()