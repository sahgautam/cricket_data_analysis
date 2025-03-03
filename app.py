from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ✅ Check if 'pipe.pkl' exists before loading
model_path = "pipe.pkl"
if os.path.exists(model_path):
    pipe = pickle.load(open(model_path, "rb"))
else:
    pipe = None  # Handle case where model is missing

# ✅ Teams and cities list for dropdowns
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            city = request.form['city']
            current_score = int(request.form['current_score'])
            overs = float(request.form['overs'])
            wickets = int(request.form['wickets'])
            last_five = int(request.form['last_five'])

            # ✅ Input validation
            if overs < 5:
                error_message = "Overs should be greater than 5 for accurate predictions."
            elif wickets > 10 or wickets < 0:
                error_message = "Wickets should be between 0 and 10."
            else:
                # ✅ Calculate additional features
                balls_left = 120 - (overs * 6)
                wicket_left = 10 - wickets  # Fixed column name
                current_run_rate = current_score / overs  # Fixed column name

                # ✅ Ensure the model is available
                if pipe is None:
                    error_message = "Model file (pipe.pkl) not found. Please check the server setup."
                else:
                    # ✅ Prepare input DataFrame
                    input_df = pd.DataFrame({
                        'batting_team': [batting_team], 
                        'bowling_team': [bowling_team], 
                        'city': [city],
                        'current_score': [current_score], 
                        'balls_left': [balls_left],
                        'wicket_left': [wicket_left],  # Fixed column name
                        'current_run_rate': [current_run_rate],  # Fixed column name
                        'last_five': [last_five]
                    })

                    # ✅ Make prediction
                    result = pipe.predict(input_df)
                    prediction = int(result[0])

        except ValueError:
            error_message = "Invalid input! Please enter correct numeric values."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction, error_message=error_message)

# ✅ Run Flask server properly
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
