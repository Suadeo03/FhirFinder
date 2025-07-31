# services/profile_training_tracker.py
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, asc
import logging

from models.database.models import Profile
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from config.database import get_db

logger = logging.getLogger(__name__)

class ProfileTrainingTracker:
    """
    Captures and tracks training scores for FHIR profiles over time
    """
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "scores").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
    
    def capture_profile_scores(self, db: Session, profile_ids: List[str] = None, 
                             days_back: int = 30) -> Dict[str, Any]:
        """
        Capture current training scores for specified profiles
        
        Args:
            db: Database session
            profile_ids: List of specific profile IDs to track (None = all profiles)
            days_back: Number of days of feedback history to consider
        
        Returns:
            Dictionary with captured scores and metadata
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Build query for profiles
            query = db.query(Profile)
            if profile_ids:
                query = query.filter(Profile.id.in_(profile_ids))
            
            profiles = query.all()
            
            captured_scores = []
            
            for profile in profiles:
                score_data = self._calculate_profile_training_score(
                    db, profile, start_date, end_date
                )
                captured_scores.append(score_data)
            
            # Create capture record
            capture_record = {
                "capture_timestamp": end_date.isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "days_analyzed": days_back,
                "total_profiles": len(profiles),
                "profiles_with_feedback": len([s for s in captured_scores if s['total_feedback'] > 0]),
                "scores": captured_scores
            }
            
            # Save to file
            self._save_capture_record(capture_record)
            
            logger.info(f"Captured training scores for {len(profiles)} profiles")
            return capture_record
            
        except Exception as e:
            logger.error(f"Error capturing profile scores: {e}")
            return {"error": str(e)}
    
    def _calculate_profile_training_score(self, db: Session, profile: Profile, 
                                        start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Calculate comprehensive training score for a single profile
        """
        try:
            # Get feedback statistics
            feedback_stats = db.query(
                func.count(UserFeedback.id).label('total_feedback'),
                func.sum(func.case([(UserFeedback.feedback_type == 'positive', 1)], else_=0)).label('positive_feedback'),
                func.sum(func.case([(UserFeedback.feedback_type == 'negative', 1)], else_=0)).label('negative_feedback'),
                func.sum(func.case([(UserFeedback.feedback_type == 'neutral', 1)], else_=0)).label('neutral_feedback'),
                func.avg(UserFeedback.original_score).label('avg_original_score')
            ).filter(
                and_(
                    UserFeedback.profile_id == profile.id,
                    UserFeedback.created_at >= start_date,
                    UserFeedback.created_at <= end_date
                )
            ).first()
            
            # Get quality metrics
            quality_metrics = db.query(SearchQualityMetrics).filter(
                and_(
                    SearchQualityMetrics.profile_id == profile.id,
                    SearchQualityMetrics.last_updated >= start_date
                )
            ).all()
            
            # Calculate scores
            total_feedback = feedback_stats.total_feedback or 0
            positive_feedback = feedback_stats.positive_feedback or 0
            negative_feedback = feedback_stats.negative_feedback or 0
            neutral_feedback = feedback_stats.neutral_feedback or 0
            
            # Satisfaction rate
            satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0.0
            
            # Training effectiveness score
            training_score = self._calculate_training_effectiveness(
                positive_feedback, negative_feedback, total_feedback
            )
            
            # Profile performance trend
            trend = self._calculate_performance_trend(db, profile.id, start_date, end_date)
            
            # Query diversity (how many different queries this profile was relevant for)
            unique_queries = db.query(func.count(func.distinct(UserFeedback.query_normalized))).filter(
                and_(
                    UserFeedback.profile_id == profile.id,
                    UserFeedback.created_at >= start_date,
                    UserFeedback.created_at <= end_date
                )
            ).scalar() or 0
            
            return {
                "profile_id": profile.id,
                "profile_name": getattr(profile, 'name', 'Unknown'),
                "resource_type": getattr(profile, 'resource_type', 'Unknown'),
                "oid": getattr(profile, 'oid', 'Unknown'),
                "capture_timestamp": datetime.utcnow().isoformat(),
                
                # Feedback metrics
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "neutral_feedback": neutral_feedback,
                "satisfaction_rate": round(satisfaction_rate, 2),
                
                # Training metrics
                "training_effectiveness_score": round(training_score, 3),
                "avg_original_similarity_score": round(feedback_stats.avg_original_score or 0.0, 3),
                "unique_queries_count": unique_queries,
                "performance_trend": trend,
                
                # Quality metrics summary
                "quality_metrics_count": len(quality_metrics),
                "avg_confidence_score": round(
                    sum(qm.confidence_score or 0 for qm in quality_metrics) / max(len(quality_metrics), 1), 3
                ),
                "avg_feedback_ratio": round(
                    sum(qm.feedback_ratio or 0 for qm in quality_metrics) / max(len(quality_metrics), 1), 3
                ),
                
                # Metadata
                "last_feedback_date": self._get_last_feedback_date(db, profile.id),
                "is_active": total_feedback > 0,
                "needs_attention": satisfaction_rate < 50.0 and total_feedback >= 5
            }
            
        except Exception as e:
            logger.error(f"Error calculating training score for profile {profile.id}: {e}")
            return {
                "profile_id": profile.id,
                "error": str(e),
                "capture_timestamp": datetime.utcnow().isoformat()
            }
    
    def _calculate_training_effectiveness(self, positive: int, negative: int, total: int) -> float:
        """
        Calculate a training effectiveness score based on feedback patterns
        
        Score factors:
        - High positive feedback = high score
        - Consistency in feedback = higher score
        - Volume of feedback = confidence multiplier
        """
        if total == 0:
            return 0.0
        
        # Basic satisfaction rate
        satisfaction = positive / total
        
        # Confidence based on volume (sigmoid function)
        volume_confidence = min(total / 20.0, 1.0)  # Max confidence at 20+ feedback items
        
        # Consistency bonus (penalize mixed signals)
        if total >= 5:
            consistency = max(positive, negative) / total  # Higher when feedback is consistent
        else:
            consistency = 0.5  # Neutral for low volume
        
        # Combined score
        effectiveness = (satisfaction * 0.6 + consistency * 0.4) * volume_confidence
        
        return min(effectiveness, 1.0)
    
    def _calculate_performance_trend(self, db: Session, profile_id: str, 
                                   start_date: datetime, end_date: datetime) -> str:
        """
        Calculate if profile performance is improving, declining, or stable
        """
        try:
            # Split period in half
            mid_date = start_date + (end_date - start_date) / 2
            
            # First half feedback
            first_half = db.query(
                func.count(UserFeedback.id).label('total'),
                func.sum(func.case([(UserFeedback.feedback_type == 'positive', 1)], else_=0)).label('positive')
            ).filter(
                and_(
                    UserFeedback.profile_id == profile_id,
                    UserFeedback.created_at >= start_date,
                    UserFeedback.created_at < mid_date
                )
            ).first()
            
            # Second half feedback
            second_half = db.query(
                func.count(UserFeedback.id).label('total'),
                func.sum(func.case([(UserFeedback.feedback_type == 'positive', 1)], else_=0)).label('positive')
            ).filter(
                and_(
                    UserFeedback.profile_id == profile_id,
                    UserFeedback.created_at >= mid_date,
                    UserFeedback.created_at <= end_date
                )
            ).first()
            
            # Calculate satisfaction rates
            first_satisfaction = (first_half.positive / first_half.total) if first_half.total > 0 else 0
            second_satisfaction = (second_half.positive / second_half.total) if second_half.total > 0 else 0
            
            # Determine trend
            if first_half.total == 0 and second_half.total == 0:
                return "no_data"
            elif first_half.total == 0:
                return "new"
            elif second_half.total == 0:
                return "inactive"
            else:
                difference = second_satisfaction - first_satisfaction
                if difference > 0.1:
                    return "improving"
                elif difference < -0.1:
                    return "declining"
                else:
                    return "stable"
                    
        except Exception as e:
            logger.error(f"Error calculating trend for profile {profile_id}: {e}")
            return "unknown"
    
    def _get_last_feedback_date(self, db: Session, profile_id: str) -> Optional[str]:
        """Get the date of the last feedback for this profile"""
        try:
            last_feedback = db.query(func.max(UserFeedback.created_at)).filter(
                UserFeedback.profile_id == profile_id
            ).scalar()
            
            return last_feedback.isoformat() if last_feedback else None
            
        except Exception as e:
            logger.error(f"Error getting last feedback date for profile {profile_id}: {e}")
            return None
    
    def _save_capture_record(self, capture_record: Dict[str, Any]):
        """Save capture record to multiple formats"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.output_dir / "scores" / f"profile_scores_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(capture_record, f, indent=2, default=str)
        
        # Save as CSV
        csv_file = self.output_dir / "scores" / f"profile_scores_{timestamp}.csv"
        df = pd.DataFrame(capture_record['scores'])
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved capture record to {json_file} and {csv_file}")
    
    def generate_training_report(self, db: Session, days_back: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive training report
        """
        try:
            # Capture current scores
            capture_data = self.capture_profile_scores(db, days_back=days_back)
            
            if "error" in capture_data:
                return capture_data
            
            scores = capture_data['scores']
            
            # Analyze the data
            analysis = {
                "summary": {
                    "total_profiles_analyzed": len(scores),
                    "profiles_with_feedback": len([s for s in scores if s['total_feedback'] > 0]),
                    "avg_satisfaction_rate": round(
                        sum(s['satisfaction_rate'] for s in scores if s['total_feedback'] > 0) / 
                        max(len([s for s in scores if s['total_feedback'] > 0]), 1), 2
                    ),
                    "total_feedback_collected": sum(s['total_feedback'] for s in scores),
                },
                
                "top_performers": sorted(
                    [s for s in scores if s['total_feedback'] >= 5],
                    key=lambda x: x['training_effectiveness_score'],
                    reverse=True
                )[:10],
                
                "need_attention": [
                    s for s in scores 
                    if s.get('needs_attention', False)
                ],
                
                "trending_up": [
                    s for s in scores 
                    if s.get('performance_trend') == 'improving'
                ],
                
                "trending_down": [
                    s for s in scores 
                    if s.get('performance_trend') == 'declining'
                ],
                
                "new_profiles": [
                    s for s in scores 
                    if s.get('performance_trend') == 'new'
                ],
                
                "inactive_profiles": [
                    s for s in scores 
                    if s.get('performance_trend') == 'inactive'
                ]
            }
            
            # Save report
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / "reports" / f"training_report_{timestamp}.json"
            
            full_report = {
                "report_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "period_analyzed": f"{days_back} days",
                    "report_type": "training_effectiveness_analysis"
                },
                "analysis": analysis,
                "raw_data": capture_data
            }
            
            with open(report_file, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            
            logger.info(f"Generated training report: {report_file}")
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating training report: {e}")
            return {"error": str(e)}
    
    def export_for_ml_training(self, db: Session, format: str = "csv", 
                              min_feedback: int = 5) -> str:
        """
        Export profile training data in format suitable for ML training
        
        Args:
            db: Database session
            format: 'csv' or 'parquet'
            min_feedback: Minimum feedback count to include profile
        
        Returns:
            Path to exported file
        """
        try:
            # Get profiles with sufficient feedback
            capture_data = self.capture_profile_scores(db, days_back=90)  # Longer period for ML
            scores = [s for s in capture_data['scores'] if s['total_feedback'] >= min_feedback]
            
            # Create ML-ready dataset
            ml_data = []
            for score in scores:
                ml_data.append({
                    'profile_id': score['profile_id'],
                    'resource_type': score['resource_type'],
                    'training_effectiveness_score': score['training_effectiveness_score'],
                    'satisfaction_rate': score['satisfaction_rate'],
                    'total_feedback': score['total_feedback'],
                    'positive_feedback_ratio': score['positive_feedback'] / max(score['total_feedback'], 1),
                    'unique_queries_count': score['unique_queries_count'],
                    'avg_confidence_score': score['avg_confidence_score'],
                    'avg_feedback_ratio': score['avg_feedback_ratio'],
                    'is_improving': 1 if score['performance_trend'] == 'improving' else 0,
                    'needs_attention': 1 if score.get('needs_attention', False) else 0,
                    'capture_timestamp': score['capture_timestamp']
                })
            
            # Export
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                export_file = self.output_dir / "exports" / f"ml_training_data_{timestamp}.csv"
                df = pd.DataFrame(ml_data)
                df.to_csv(export_file, index=False)
            elif format.lower() == 'parquet':
                export_file = self.output_dir / "exports" / f"ml_training_data_{timestamp}.parquet"
                df = pd.DataFrame(ml_data)
                df.to_parquet(export_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported ML training data to {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Error exporting ML training data: {e}")
            return f"Error: {e}"
    
    def track_profile_over_time(self, db: Session, profile_id: str, 
                               days_back: int = 90) -> Dict[str, Any]:
        """
        Track a specific profile's performance over time
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Get feedback over time (weekly buckets)
            weekly_data = []
            current_date = start_date
            
            while current_date < end_date:
                week_end = min(current_date + timedelta(days=7), end_date)
                
                week_stats = db.query(
                    func.count(UserFeedback.id).label('total'),
                    func.sum(func.case([(UserFeedback.feedback_type == 'positive', 1)], else_=0)).label('positive'),
                    func.avg(UserFeedback.original_score).label('avg_score')
                ).filter(
                    and_(
                        UserFeedback.profile_id == profile_id,
                        UserFeedback.created_at >= current_date,
                        UserFeedback.created_at < week_end
                    )
                ).first()
                
                satisfaction = (week_stats.positive / week_stats.total * 100) if week_stats.total > 0 else 0
                
                weekly_data.append({
                    'week_start': current_date.strftime('%Y-%m-%d'),
                    'week_end': week_end.strftime('%Y-%m-%d'),
                    'total_feedback': week_stats.total or 0,
                    'positive_feedback': week_stats.positive or 0,
                    'satisfaction_rate': round(satisfaction, 2),
                    'avg_similarity_score': round(week_stats.avg_score or 0, 3)
                })
                
                current_date = week_end
            
            # Get profile details
            profile = db.query(Profile).filter(Profile.id == profile_id).first()
            
            tracking_data = {
                "profile_id": profile_id,
                "profile_name": getattr(profile, 'name', 'Unknown') if profile else 'Unknown',
                "tracking_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "weekly_performance": weekly_data,
                "overall_trend": self._calculate_performance_trend(db, profile_id, start_date, end_date),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Save tracking data
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            tracking_file = self.output_dir / "reports" / f"profile_tracking_{profile_id}_{timestamp}.json"
            
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2, default=str)
            
            logger.info(f"Generated profile tracking report: {tracking_file}")
            return tracking_data
            
        except Exception as e:
            logger.error(f"Error tracking profile {profile_id}: {e}")
            return {"error": str(e)}


# Usage example and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Training Score Tracker")
    parser.add_argument("--action", choices=["capture", "report", "export", "track"], 
                       required=True, help="Action to perform")
    parser.add_argument("--profile-ids", nargs="*", help="Specific profile IDs to analyze")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", 
                       help="Export format for ML data")
    parser.add_argument("--profile-id", help="Single profile ID for tracking")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ProfileTrainingTracker(output_dir=args.output_dir)
    db = next(get_db())
    
    try:
        if args.action == "capture":
            result = tracker.capture_profile_scores(db, args.profile_ids, args.days)
            print(f"Captured scores for {len(result.get('scores', []))} profiles")
            
        elif args.action == "report":
            result = tracker.generate_training_report(db, args.days)
            print(f"Generated training report with {len(result.get('analysis', {}).get('top_performers', []))} top performers")
            
        elif args.action == "export":
            result = tracker.export_for_ml_training(db, args.format)
            print(f"Exported ML training data to: {result}")
            
        elif args.action == "track":
            if not args.profile_id:
                print("Error: --profile-id required for track action")
            else:
                result = tracker.track_profile_over_time(db, args.profile_id, args.days)
                print(f"Generated tracking report for profile {args.profile_id}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()