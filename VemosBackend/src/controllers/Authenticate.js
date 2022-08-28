const ldap = require('ldapjs');
const jwt = require('jsonwebtoken');
const authConfig = require('../auth');

module.exports = {
    async authentication(req, res){
        const { login, password } = req.body;
        let dn = null;
        let cn = null;

        let client = ldap.createClient({
            url: 'ldap://192.168.0.249'
        });

        client.search('dc=labic,dc=utfpr,dc=edu,dc=br', {scope: 'sub'}, function(err, response){
            response.on('searchEntry', function(entry) {
                if(entry.object.uid === login){
                    dn = entry.object.dn
                    cn = entry.object.cn
                }
            });
            response.on('end', function(result) {
                client.bind(dn, password, function(err) {
                    if(err !== null){
                        return res.send({ error: 'Invalid Credentials'});
                    }
                    else{
                        const token = jwt.sign({ id: cn }, authConfig.secret);
                        return res.send({ cn, login, token});
                    }
                });
            });
            response.on('error', function(err) {
                console.log(err)
                return res.send({ error: 'Problem on the server'});
            });
        })
    }
}