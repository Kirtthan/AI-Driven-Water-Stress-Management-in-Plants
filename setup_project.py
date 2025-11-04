import os

# Create necessary directories
directories = ['models', 'data', 'assets']

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Created directory: {directory}")
    else:
        print(f"ℹ️ Directory already exists: {directory}")

print("\n✅ Project structure setup complete!")
print("\nNext steps:")
print("1. Run: python enhanced_model_training.py")
print("2. Run: streamlit run enhanced_app.py")