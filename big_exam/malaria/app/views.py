from django.shortcuts import render
import joblib
import os

# Create your views here.


def home(request):
    return render(request,'index.html')

def results(request):
    data=[]
    data.append(request.GET["temperature"])
    data.append(request.GET["parasite_density"])
    data.append(request.GET["wbc_count"])
    data.append(request.GET["hb_level"])
    data.append(request.GET["hematocrit"])
    data.append(request.GET["mean_cell_volume"])
    data.append(request.GET["mean_corp_hb"])
    data.append(request.GET["mean_cell_hb_conc"])
    data.append(request.GET["platelet_count"])
    data.append(request.GET["platelet_distr_width"])
    data.append(request.GET["mean_platelet_vl"])
    data.append(request.GET["neutrophils_percent"])
    data.append(request.GET["lymphocytes_percent"])
    data.append(request.GET["mixed_cells_percent"])
    data.append(request.GET["neutrophils_count"])
    data.append(request.GET["lymphocytes_count"])
    data.append(request.GET["mixed_cells_count"])
    model=joblib.load(os.path.join(os.path.dirname(__file__),"malaria_model.joblib"))
    diagnosis=model.predict([data])
    return render(request,'results.html',{'diagnosis':diagnosis})