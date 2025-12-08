import os
import sys
import streamlit as st
from tempfile import NamedTemporaryFile

def main():
    try:
        # Get the code from secrets
        code = os.environ.get("MAIN_CODE")
        
        if not code:
            st.error("⚠️ The application code wasn't found in secrets. Please add the MAIN_CODE secret.")
            return
        
        # Create a temporary Python file
        with NamedTemporaryFile(suffix='.py', delete=False, mode='w') as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        # Execute the code
        exec(compile(code, tmp_path, 'exec'), globals())
        
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
            
    except Exception as e:
        st.error(f"⚠️ Error loading or executing the application: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()