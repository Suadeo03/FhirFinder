from sqlalchemy.orm import Session
from models.database.models import QueryPerformance

# Records the query and sources of the matches

def create_performance_log(profile_id, query_text, profile_name, profile_oid, 
                           profile_score, context_score, combined_score, 
                           match_reasons, keywords, db: Session):

    try:
        query_perf = QueryPerformance(
            profile_id=profile_id,
            query_text=query_text,
            profile_name=profile_name,
            profile_oid=profile_oid,
            profile_score=profile_score,
            context_score=context_score,
            combined_score=combined_score,
            match_reasons=match_reasons,
            keywords=keywords
        )
        
        db.add(query_perf)
        db.commit()
        db.refresh(query_perf)
        return query_perf
        
    except Exception as e:
        Session.rollback()
        raise e
