import google.generativeai as genai

genai.configure(api_key="AIzaSyBIFD9H_0Dskw5NQg-EsOdnSIiM8Eo6VSc")

models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
