import streamlit as st
import program
import matplotlib.pyplot as plt

def main():
    
    st.title('City Travel Analyzer')

    # Add input elements
    input_data = st.text_input('Enter Input Data')

    # Add a button to trigger the processing
    if st.button('Process'):
        # Call your_program_function with the input data
        output_data = program.your_program_function(input_data)
        
        # Display the output data
        st.subheader('Output Data:')
        st.dataframe(output_data)
        
    # Add buttons for different graph types
    if st.button('Show Bar Graph'):
        fig1 = program.bar_func(input_data)
        st.subheader('Bar Graph')
        st.pyplot(fig1)
        
    if st.button('Show Line Graph'):
        fig2 = program.line_func(input_data)
        st.subheader('Line Graph')
        st.pyplot(fig2)
        
    if st.button('Visualize Metagraph'):
        fig3 = program.metagraph_func(input_data)
        st.subheader('Metagraph Visualization')
        st.pyplot(fig3)

# Run the main function to start the Streamlit application
if __name__ == '__main__':
    main()
#
#C:/Users/athar/Downloads/hello/dataz.csv