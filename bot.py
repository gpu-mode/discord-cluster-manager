from dotenv import load_dotenv
from github import Github
import os
import time
from datetime import datetime, timezone
import requests
import logging
import traceback
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

if not os.getenv('DISCORD_TOKEN'):
    logger.error("DISCORD_TOKEN not found in environment variables")
    raise ValueError("DISCORD_TOKEN not found")
if not os.getenv('GITHUB_TOKEN'):
    logger.error("GITHUB_TOKEN not found in environment variables")
    raise ValueError("GITHUB_TOKEN not found")
if not os.getenv('GITHUB_REPO'):
    logger.error("GITHUB_REPO not found in environment variables")
    raise ValueError("GITHUB_REPO not found")

logger.info(f"Using GitHub repo: {os.getenv('GITHUB_REPO')}")

def trigger_github_action():
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    try:
        trigger_time = datetime.now(timezone.utc)
        
        workflow = repo.get_workflow("train_workflow.yml")
        response = workflow.create_dispatch("main")
        
        logger.info(f"Workflow found: {workflow}")
        logger.info(f"Workflow dispatch response: {response}")

        if response:
            time.sleep(2)
            runs = list(workflow.get_runs())
            for run in runs:
                logger.info(f"Checking workflow run with ID: {run.id}, created at: {run.created_at}")
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    logger.info(f"Found matching workflow run with ID: {run.id}")
                    return run.id
        
        logger.warning("Workflow dispatch failed, check permissions.")
        return None
    except Exception as e:
        logger.error(f"Error triggering GitHub Action: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def download_artifact(run_id):
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        run = repo.get_workflow_run(run_id)
        logger.info(f"Fetching artifacts for run ID: {run_id}")
        
        artifacts = run.get_artifacts()
        logger.info(f"Found {artifacts.totalCount} artifacts")
        
        for artifact in artifacts:
            logger.info(f"Found artifact: {artifact.name}")
            if artifact.name == 'training-logs':
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully downloaded artifact")
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile('training.log.zip') as z:
                        with z.open('training.log') as f:
                            logs = f.read().decode('utf-8')
                    
                    os.remove('training.log.zip')
                    return logs
                else:
                    logger.error(f"Failed to download artifact. Status code: {response.status_code}")
        
        logger.warning("No training-logs artifact found")
        return "No training logs found in artifacts"
    except Exception as e:
        logger.error(f"Error downloading artifact: {str(e)}")
        logger.debug(traceback.format_exc())
        return f"Error downloading artifacts: {str(e)}"

def check_workflow_status(run_id):
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        try:
            run = repo.get_workflow_run(run_id)
            logger.info(f"Current status of run ID {run_id}: {run.status}")
            
            if run.status == "completed":
                logger.info("Workflow completed, downloading artifacts")
                logs = download_artifact(run_id)
                return run.conclusion, logs, run.html_url
            
            logger.info(f"Workflow still running... Status: {run.status}")
            logger.info(f"Live view: {run.html_url}")
            time.sleep(30)
        except Exception as e:
            logger.error(f"Error checking workflow status: {str(e)}")
            logger.debug(traceback.format_exc())
            return "error", str(e), None

if __name__ == "__main__":
    run_id = trigger_github_action()
    
    if run_id:
        logger.info(f"GitHub Action triggered successfully! Run ID: {run_id}")
        logger.info("Monitoring progress...")
        
        status, logs, url = check_workflow_status(run_id)
        
        logger.info(f"\nWorkflow completed with status: {status}")
        logger.info(f"\nTraining Logs:\n{logs}")
        logger.info(f"\nView the full run at: {url}")
    else:
        logger.error("Failed to trigger GitHub Action. Please check your configuration.")