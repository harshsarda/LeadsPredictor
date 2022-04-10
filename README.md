# LeadsPredictor
This repository contains a leads prediction model for a sample job marketplace data 

Instructions:

1. Run train.py. This will create & dump model and feature transformers objects to the required directories.
2. cd to LeadsPredictor directory.
3. run the command: docker-compose up --build -d
4. Test the API using API Testing jupyter notebook in the notebooks folder
5. For FastAPI UI, you can go to this link and perform a sample test as well: http://localhost:8087/docs#/default/get_leads_for_job_posting_items__post

Notes:
1. Request Body format can be seen from main.py
2. Response contains of Payload which would contain the n predictions (n depends on input through the request) in the form of a list and total which contains n

