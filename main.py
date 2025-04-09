from flow.write_survey_outline_flow import WriteSurveyOutlineFlow

def kickoff():
    flow = WriteSurveyOutlineFlow()
    flow.kickoff()  # Runs the full step-by-step pipeline

if __name__ == "__main__":
    kickoff()
