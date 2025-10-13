let Userlocation = navigator.geolocation;
const tempinp = document.getElementById("temp");
const humidinp = document.getElementById("humidity");
const rainfallinp = document.getElementById("rainfall");
const zipcodeinp = document.getElementById("zipcode");
const soilpHinp = document.getElementById("pH");
const labelzip = document.getElementById("label_for_zipcode");

const getZipCode = async (lat,lon) => {
    fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`).then(async data => {
        let res = await data.json();
        zipcodeinp.value = res.address.postcode;
    })
}

const TempAndHum = (lat, lon) => {
    fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=56daa81639b1d12d9d66df949ad81988`).then(async data => {
        let res = await data.json();
        const { temp, humidity } = res.main;
        
        tempinp.value = (temp-273.15).toFixed(0);
        humidinp.value = humidity;
        
    }).catch(() => {
        console.log("error in fetching data.");
        
    });
}

const RainfallandsoilData = (lat,lon) => {
    fetch(`https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=2024-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=Asia/Kolkata`).then(async data => {
        let res = await data.json();
        let precipitation_sum = res.daily.precipitation_sum;
        let rainfall = 0;
        for (let i = 0; i < precipitation_sum.length; i++ ) {
            rainfall += precipitation_sum[i];
        }
        try{
            const soilpH = await fetch("http://127.0.0.1:5001/get-soilpH", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat, lon})
            }).then(async response => {
                const dataSoil = await response.json();
                if(dataSoil[0].Soilph !== "error in fetching soil data"){
                    soilpHinp.value = dataSoil[0].Soilph;
                    soilpHinp.hidden = true;
                }
            
            });
        } catch (err) {
            console.error(err);
        }

        rainfallinp.value = (rainfall/12).toFixed(0);
    }).catch(() => {
        console.log("error");
        
    })
}

zipcodeinp.addEventListener("input", () => {
    fetch(`https://api.postalpincode.in/pincode/${zipcodeinp.value}`)
    .then(async data => {
        let result = await data.json();
        const city = result[0].PostOffice[0].Block.split(" ")[0];
    

        fetch(`https://nominatim.openstreetmap.org/search?q=${city}&format=json`)
        .then(async data => {
            const result = await data.json();
            const {lat,lon} = result[0];
            
            await TempAndHum(lat,lon);
            await RainfallandsoilData(lat, lon);
        })
    })
})
    


const showlocation = async (pos) => {
    await TempAndHum(pos.coords.latitude, pos.coords.longitude);
    await RainfallandsoilData(pos.coords.latitude, pos.coords.longitude);
    await getZipCode(pos.coords.latitude, pos.coords.longitude);
    labelzip.innerText = "Zipcode (If wrong, change to your zip code and press ENTER)";
}

Userlocation.getCurrentPosition(showlocation);

