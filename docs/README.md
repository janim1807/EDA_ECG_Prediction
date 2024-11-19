# Experimental Stock Market Study

## Overview

In this study, participants interact in an experimental stock market created using the oTree platform. The goal is to understand investment decisions and physiological responses during trading. The experiment involves multiple rounds of investment decisions, feedback, and summaries, while collecting biosensor data to monitor physiological states.

## Data Collected

### Investment Rounds (All_Apps_Wide)
- **Investment Decisions**:
  - Participants make buy, sell, or hold decisions across 50 trading rounds.
  - Each round contains 5 decision pages with prices ranging from -25% to +25% of the current market price.
  
- **Feedback and Wallet Summary**:
  - After each round, participants receive feedback on their transactions.
  - Participants can view a summary of their current stock and cash holdings.
  - Real-time conversion to euros is provided based on the final wallet value.
  
- **Group Interaction**:
  - Participants wait for others to finish decisions each round.
  - Stocks are bought or sold based on collective market prices influenced by all participants' choices.

### Biosensor Data

#### EDA / GSR / SC
- **Measure**: Electrical conductivity of the skin (sweat gland activity).
- **Use**: Paired with other biometric sensors (eye tracking, EEG, ECG, etc.).
- **Benefits**: Provides real-time data on physiological responses; non-invasive.
- **Challenges**: Does not distinguish emotions; influenced by physical movements or environmental factors.

#### ECG
- **Measure**: Heart's electrical activity.
- **Metrics**: Heart rate, inter-beat interval, heart rate variability (HRV).
- **Relevance**: HRV linked to psychological states; explores heart-brain connection.
- **Benefits**: Affordable, non-invasive.
- **Challenges**: Crucial and complex data collection.