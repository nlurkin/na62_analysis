access_token=$(cat .token)
project_id=371

ret=$(curl --header "PRIVATE-TOKEN: ${access_token}" "https://cp3-git.irmp.ucl.ac.be/api/v4/projects/${project_id}/pipelines/latest")

curl --location --header "PRIVATE-TOKEN: ${access_token}" "https://cp3-git.irmp.ucl.ac.be/api/v4/projects/${project_id}/jobs/artifacts/main/download?job=pages" --output pages.zip
unzip pages.zip
mv public doc_compiled
rm pages.zip

echo "================================================================="
echo "Documentation has now been extracted to doc_compiled directory"
echo "You can now browse it by running (use your favourite browser):"
echo "firefox doc_compiled/index.html &"
echo "================================================================="