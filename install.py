import launch

if not launch.is_installed("google-generativeai"):
    launch.run_pip("install google-generativeai", "requirements for Promptgen")

if not launch.is_installed("python-dotenv"):
    launch.run_pip("install python-dotenv", "requirements for Promptgen")
