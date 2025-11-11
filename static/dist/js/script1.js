let Userlocation = navigator.geolocation;
const zipcodeinp = document.getElementById("zipcode");

const fillValue = (pincode) =>{
    fetch(`https://api.postalpincode.in/pincode/${pincode}`)
    .then(async data => {
        let result = await data.json();
        const city = result[0].PostOffice[0].Block.split(" ")[0];
        const district = result[0].PostOffice[0].District;
        const state = result[0].PostOffice[0].State;
        const country =result[0].PostOffice[0].Country;

        document.getElementById("city").value = city;
        document.getElementById("district").value = district;
        document.getElementById("state").value = state;
        document.getElementById("country").value = country;
    })
}

const getZipCode = async (lat,lon) => {
    fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`).then(async data => {
        let res = await data.json();
        zipcodeinp.value = res.address.postcode;
        fillValue(res.address.postcode);
    })
}

zipcodeinp.addEventListener("input", () => {
    fillValue(zipcodeinp.value);
})

const showlocation = async (pos) => {
    await getZipCode(pos.coords.latitude, pos.coords.longitude);
    labelzip.innerText = "Zipcode (If wrong, change to your zip code and press ENTER)";
}

Userlocation.getCurrentPosition(showlocation);

