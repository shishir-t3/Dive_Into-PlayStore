# A Dive Into Google Play Store

This Project involves collection of App data from the google playstore, using Google Play Scraper API based on their respective Categories and Countries.
App reviews data for unique users was also scraped and cleaned, to build a recommendation system. The model was based on a deep learning Neural network framework made by Tensorflow functional API to create user and app embeddings. It also incorporated the app-features for the rated apps, to make it a more Hybrid collaborative filtering model.

The model was deployed using Streamlit, and can be accessed at -- https://freakyeinstein-dive-into-playstore-deploy-ki4ke0.streamlitapp.com/
Clear cache before you start using the application.

All the files in the assets folder are fetched from various sources, cleaned and pre-processed to input them into our training model.
You can clone the repository and follow these steps if you want to make changes for the application and keep it working 

- clone it in a directory
- use the command `pipenv shell`.
- use the command `pipenv install`.
- once the virtual environment is ready use the command `streamlit run deploy.py` to run on the local machine.
